import numpy as np
from scipy.integrate import quad
from typing import Callable, Tuple, Dict, Optional
from ..BCs.boundary_conditions import BCs_1d


def stiffness_matrix_1d(
        x_nodes: np.ndarray,
        eps: float,
        beta: float,
        phi: Callable[[float, float], float],
        phi_prime: Callable[[float, float], float]
) -> np.ndarray:
    """
    Assemble the stiffness matrix for a 1D finite element method on a non-uniform mesh applied
    to the convection-diffusion equation.

    This function computes the global stiffness matrix for a 1D convection-diffusion problem using
    linear finite elements. The matrix is assembled by integrating contributions from the element-wise
    stiffness matrices, which are computed using the given basis functions and their derivatives.

    Parameters
    ----------
    x_nodes : np.ndarray
        A 1D array of node coordinates in the 1D mesh. Must be ordered, non-uniform meshes are supported.
    eps : float
        The diffusion coefficient.
    beta : float
        The convection coefficient.
    phi : Callable[[float, float], float]
        The basis function. Should take arguments `(x, x_i)` where `x` is the evaluation point(s)
        and `x_i` is the node associated with the basis function.
    phi_prime : Callable[[float, float], float]
        The derivative of the basis function. Should take arguments `(x, x_i)` where `x` is the evaluation
        point and `x_i` is the node associated with the basis function.

    Returns
    -------
    A : np.ndarray
        The assembled global stiffness matrix as ndarray of shape (N+1, N+1), where `N` is the number
        of elements (equal to `len(x_nodes) - 1`). Each entry corresponds to the integral of weighted
        products of basis functions and their gradients over the elements.

    Notes
    -----
    The stiffness matrix is computed using numerical integration (via `scipy.integrate.quad`)
    over each element, which accounts for potentially non-uniform spacing between nodes.
    """
    N = len(x_nodes) - 1        # Number of elements
    A = np.zeros((N+1, N+1))    # Initialise the global stiffness matrix

    # Loop over each element
    for e in range(len(x_nodes) - 1):
        # Define the boundaries of the current element
        x_left, x_right = x_nodes[e], x_nodes[e + 1]

        # Initialise the element stiffness matrix for the current element
        K_e = np.zeros((2, 2))  # 2x2 for linear basis functions (two nodes per element)

        # Loop over local basis functions (i: row, j: column in `K_e`)
        for i in range(2):  # Local basis function i (0: left node, 1: right node)
            for j in range(2):  # Local basis function j (0: left node, 1: right node)
                # Define the nodes associated with basis functions i and j
                x_i = x_nodes[e + i]
                x_j = x_nodes[e + j]

                # Compute the diffusion contribution
                diff_integrand = lambda x: (eps * phi_prime(x, x_i) *
                                            phi_prime(x, x_j))
                diff_contribution, _ = quad(diff_integrand, x_left, x_right)

                # Compute the convection contribution
                conv_integrand = lambda x: (beta * phi_prime(x, x_i) *
                                             phi(x, x_j))
                conv_contribution, _ = quad(conv_integrand, x_left, x_right)

                # Sum contributions to form the local stiffness matrix entry
                K_e[j, i] = diff_contribution + conv_contribution

        # Assemble the local stiffness matrix `K_e` into the global stiffness matrix `A`
        A[e:e + 2, e:e + 2] += K_e

    return A


def load_vector_1d(
        f: Callable[[float], float],
        epsilon: float,
        beta: float,
        x_nodes: np.ndarray,
        phi: Callable[[float, float], float],
        phi_prime: Callable[[float, float], float],
        bctype: Optional[Tuple[int, int]] = (0, 0),
        bcs: Optional[Tuple[float, float]] = (0.0, 0.0)
) -> np.ndarray:
    """
    Assemble the right-hand side (RHS) vector for a 1D finite element convection-diffusion problem.

    This function computes the RHS vector for the linear system arising from the weak fromulation
    of a 1D PDE. The computation incorporates the source term, boundary conditions, and the basis
    functions used in the finite element method.

    Parameters
    ----------
    f : Callable[[float], float]
        The source term function, f(x), defined over the domain.
    epsilon : float
        Diffusion coefficient.
    beta : float
        Convection coefficient.
    x_nodes : np.ndarray
        The nodal coordinates of the mesh.
    phi : Callable[[float, float], float]
        The basis funcion, defined as phi(x, x_i) for a given node x_i.
    phi_prime : Callable[[float, float], float]
        The derivative of the basis function.
    bctype : tuple[int, int], optional
        A 2-tuple specifying the type of boundary conditions:
        - (0, 0): Dirichlet at both boundaries.
        - (0, 1): Dirichlet at the left boundary, Neumann at the right.
        - (1, 0): Neumann at the left boundary, Dirichlet at the right.
        Default is (0, 0).
    bcs : tuple[float, float], optional
        A 2-tuple specifying the boundary values:
        - For Dirichlet conditions: (g0, g1) where u(0) = g0 and u(1) = g1.
        - For Neumann conditions: (q0, q1) where u"(0) = q0 and u"(1) = q1.
        Default is (0, 0).

    Returns
    -------
    F : np.ndarray
        The assembled RHS vector as ndarray with shape (len(x_nodes),). Each entry corresponds to the
        contribution of the respective basis function.

    Notes
    -----
    - The function uses numerical integration (via `scipy.integrate.quad`) to compute
    element contributions to the RHS vector.
    - Dirichlet boundary conditions are incorporated into the formulation through the
      `BCs_1d()` function, which provides `u_g` and `u_g_prime`.
    - Neumann boundary conditions directly modify the entries of the RHS vector.
    """
    # Initialise the RHS vector with zeros
    F = np.zeros(len(x_nodes))

    # Extract boundary condition values
    g0, g1 = bcs[0], bcs[1]

    # Compute interpolated boundary functions u_g(x) and u_g_prime(x)
    u_g, u_g_prime = BCs_1d(g0, g1, bctype, domain=(x_nodes[0], x_nodes[-1]))

    # Loop over nodes to compute the contribution of each basis function
    for i, x_i in enumerate(x_nodes):
        # Determine the integration limits for the current node
        left = x_nodes[np.maximum(i-1, 0)]
        right = x_nodes[np.minimum(i+1, len(x_nodes) - 1)]

        # Integrate contributions from source term, diffusion, and convection
        F[i], _ = quad(
            lambda x: f(x) * phi(x, x_i) -      # Source term
                      epsilon * u_g_prime(x) * phi_prime(x, x_i) -  # Diffusion term
                      beta * u_g_prime(x) * phi(x, x_i),            # Convection term
            left,
            right
        )

    # Apply modifications for Neumann boundary conditions, if present
    if bctype == (0, 1):
        F[-1] += epsilon * g1
    elif bctype == (1, 0):
        F[0] -= epsilon * g0

    # Return the assembled RHS vector
    return F
