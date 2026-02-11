import numpy as np
from scipy.linalg import solve
from typing import Tuple, Callable
from scipy.integrate import solve_bvp
from ..mixins_1d.basis_functions import BasisFunctionMixin
from ..mixins_1d.mesh_refinement import AdaptiveRefinementMixin
from ..assembly.linear_system_assembly_1d import stiffness_matrix_1d, load_vector_1d


class ConvectionDiffusion_1d(BasisFunctionMixin, AdaptiveRefinementMixin):
    def __init__(self,
                 x_nodes: np.ndarray,
                 epsilon: float = 0.01,
                 beta: float = 1.0,
                 bctype: Tuple[int, int] = (0, 0),
                 bcs: Tuple[float, float] = (0, 0)
                 ) -> None:
        """
        Creates an instance of the ConvectionDiffusionFEM class, intended to apply Finite Element Methods
        to the 1-dimensional convection-diffusion equation -epsilon u''(x) + beta u'(x) = f(x).

        Parameters
        ----------
        x_nodes : ndarray
            Array containing the N+1 nodes
        epsilon : float, optional
            Diffusion coefficient.
        beta : float, optional
            Convection constant.
        uniform : bool, optional
            If True, the mesh will contain N elements of uniform size 1 / N. Else, an array of nodes
            uniformly spaced within the domain [0, 1] will be quadratically clustered towards x=1.
        bctype : tuple[int, int], optional
            Specifies the left and right type of boundary conditions:
                bctype=0 for Dirichlet BCs.
                bctype=1 for Neumann BCs.
                bctype=2 for Robin BCs.
        bcs : tuple[float, float], optional
            Specifies the left and right BCs.
        """
        self.x_nodes = x_nodes
        self.N = len(x_nodes) - 1
        self.L = x_nodes[-1] - x_nodes[0]
        self.epsilon = epsilon
        self.beta = beta
        self.bctype = bctype
        self.bcs = bcs

        if self.bctype == (1, 1):
            raise ValueError("Neumann boundary conditions cannot be applied to both boundaries, as the system is left "
                             "underdetermined. Use a Dirichlet condition on at least one boundary")

        # Generate mesh and obtain array of spacings 'self.dxs'
        indices = [i for i in range(len(x_nodes))]
        self.node_to_idx = {k: v for k, v in zip(x_nodes, indices)}
        self.dxs = x_nodes[1:] - x_nodes[:-1]

    def solve(self,
              f: Callable[[float], float]
              ) -> Callable[[float], float]:
        """
        Solve the FEM system for the 1D convection-diffusion equation with a RHS of f.
        Returns a lambda function u(x) with the FEM approximation of the solution.

        Parameters
        ----------
        f : Callable[[float], float]
            RHS of the one dimensional convection-diffusion equation to solve.

        Returns
        -------
        u_h : Callable[[float], float]
            Homogeneous part of the solution u_h(x).
        """
        F = load_vector_1d(f, self.epsilon, self.beta, self.x_nodes,
                            lambda x, x_i: self.phi_linear(x, x_i),
                            lambda x, x_i: self.phi_linear_derivative(x, x_i),
                            bctype=self.bctype, bcs=self.bcs)
        A = stiffness_matrix_1d(self.x_nodes, self.epsilon, self.beta, self.phi_linear, self.phi_linear_derivative)

        if self.bctype == (0, 0):
            F = F[1:-1]
            A = A[1:-1, 1:-1]
        elif self.bctype == (1, 0):
            F = F[:-1]
            A = A[:-1, :-1]
        elif self.bctype == (0, 1):
            F = F[1:]
            A = A[1:, 1:]

        U = solve(A, F)
        if self.bctype == (0, 0):
            U = np.concatenate(([0], U, [0]))
        elif self.bctype == (1, 0):
            U = np.concatenate((U, [0]))
        elif self.bctype == (0, 1):
            U = np.concatenate(([0], U))

        # Create the callable for the homogeneous solution u_h(x)
        u_h = lambda x: np.sum(U_i * self.phi_linear(x, x_i) for U_i, x_i in zip(U, self.x_nodes))
        return u_h

    def refine_mesh(self, f, tol=0.5, max_iter=20):
        new_nodes, u_fem, nodal_history = self.adaptive_mesh_refinement(f, tol=tol, max_iter=max_iter)
        return new_nodes, u_fem, nodal_history

    def solve_scipy(self, f, x_eval):
        """
        Solve the same problem using SciPy's solve_bvp for comparison. Accounts
        for Dirichlet, Neumann, and mixed boundary conditions.

        Parameters
        ----------
        x_eval : ndarray
            Values of x for which the solution will be computed.

        Returns
        -------
        sol.x : ndarray
            Values of x on which the solution is approximated.
        sol.y[0] : ndarray
            Values of u(x) obtained with solve_bvp().
        """

        g0, g1 = self.bcs[0], self.bcs[1]

        # Define ODE and specify its boundary conditions (BC)
        def ode_system(x, y):
            """
            Defines the ODE system:
            y[0] = u(x) is the solution.
            y[1] = u'(x) is the solution derivative.
            """
            return np.vstack((y[1], (self.beta * y[1] - f(x)) / self.epsilon))

        def bc(ya, yb):
            """
            Specifies boundary conditions based on self.bctype.
            - (0, 0): Dirichlet at both boundaries.
            - (0, 1): Dirichlet at the left boundary, Neumann at the right.
            - (1, 0): Neumann at the left boundary, Dirichlet at the right.
            """
            if self.bctype == (0, 0):
                return np.array([ya[0] - g0, yb[0] - g1])
            elif self.bctype == (0, 1):
                return np.array([ya[0] - g0, yb[1] - g1])
            elif self.bctype == (1, 0):
                return np.array([ya[1] - g0, yb[0] - g1])

        # Apply solve_bvp to the same mesh
        x = self.x_nodes
        y_initial = np.zeros((2, x.size))
        sol = solve_bvp(ode_system, bc, x, y_initial)

        # Evaluate at points in x_eval
        u_eval = sol.sol(x_eval)[0]
        return u_eval
