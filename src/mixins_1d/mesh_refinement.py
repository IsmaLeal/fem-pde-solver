import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from fem_portfolio.archive.boundary_conditions_old import BCs_1d
from ..mesh.mesh import refine_mesh_1d
from typing import Callable, Optional


class AdaptiveRefinementMixin:
    '''
    A mixin class for performing adaptive mesh refinement (AMR) in 1D FEM.

    This class provides functionality to iteratively refine the mesh based on error indicators
    computed from the FEM solution, ensuring better resolution in regions with high gradients or
    large errors.

    Methods
    -------
    adaptive_mesh_refinement(f, tol, max_iter=20)
        Performs AMR for the FEM solution.
    update_mesh_properties()
        Updates mesh-related attributes such as node spacing 'self.dxs'.

    Notes
    -----
    This mixin is designed to be inherited by a solver class providing:
    - 'self.solve(f)': A method to compute the FEM solution for a given RHS function 'f'.
    - 'self.x_nodes': An array of current mesh node coordinates.
    - 'self.bcs': Boundary condition values.
    - 'self.bctype': The type of boundary conditions.
    '''
    def adaptive_mesh_refinement(self,
                                 f: Callable[[float], float],
                                 tol: float,
                                 max_iter: Optional[int] =20
                                 ) -> np.ndarray:
        '''
        Perform adaptive mesh refinement for 1D FEM solution.

        Parameters
        ----------
        f : Callable[[float], float]
            RHS of the original ODE.
        tol : float
            Refinement tolerance.
        max_iter : int, optional
            Maximum number of refinement iterations.

        Returns
        -------
        refined_nodes : ndarray
            Refined mesh nodes.
        u_fem : Callable[[float], float]
            FEM solution on the refined mesh.
        nodal_history : list of ndarray
            The history of node positions across refinement iterations.
        '''

        u_g, _ = BCs_1d(self.bcs[0], self.bcs[1], bctype=self.bctype, domain=(self.x_nodes[0], self.x_nodes[-1]))
        u_h = self.solve(f)
        u_fem = lambda x: u_g(x) + u_h(x)

        nodal_history = [self.x_nodes.copy()]

        for _ in tqdm(range(max_iter)):
            # Evaluate u_fem on the current mesh
            u_values = u_fem(self.x_nodes)

            # Compute the error indicator (difference between adjacent FEM values)
            errors = np.abs(np.diff(u_values))

            # Obtaine refined array
            refined_nodes = refine_mesh_1d(self.x_nodes, errors, tol)

            # Update the mesh
            self.x_nodes = refined_nodes
            self.update_mesh_properties()
            u_h = self.solve(f)
            u_fem = lambda x: u_h(x) + u_g(x)

            nodal_history.append(self.x_nodes.copy())

        return self.x_nodes, u_fem, nodal_history

    def update_mesh_properties(self):
        '''
        Update mesh-related attributes after refinement.

        This method calculates:
        - 'self.dxs': The spacings between adjacent nodes in the mesh.
        - 'self.node_to_idx': A dictionary mapping node coorindates to their indices.

        Notes
        -----
        This method should be called whenever 'self.x_nodes' is modified to ensure
        consistency in posterior calculations.
        '''
        indices = [i for i in range(len(self.x_nodes))]
        self.node_to_idx = {k: v for k, v in zip(self.x_nodes, indices)}
        self.dxs = self.x_nodes[1:] - self.x_nodes[:-1]


def plot_node_refinement(nodal_history):
    # Obtain domain limits
    x_min = nodal_history[0][0]
    x_max = nodal_history[0][-1]

    # Set up the figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.1, 0.1)  # Small y-range to visualize points better
    ax.set_title("Adaptive Mesh Refinement")
    ax.set_xlabel("Domain (x)")
    ax.set_ylabel("")

    # Initialize an empty scatter plot
    scatter, = ax.plot([], [], 'o', color='blue')

    # Update function for the animation
    def update(frame):
        ax.set_title(f"Adaptive Mesh Refinement Step {frame + 1}")
        nodes = nodal_history[frame]
        scatter.set_data(nodes, [0] * len(nodes))  # Plot nodes along the x-axis
        return scatter,

    # Create the animation
    num_frames = len(nodal_history)
    ani = FuncAnimation(fig, update, frames=num_frames, interval=500, blit=True)

    # Display the animation
    #plt.show()
