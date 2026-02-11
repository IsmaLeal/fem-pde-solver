import numpy as np
from scipy.integrate import quad
from typing import Optional, Union, Tuple, Callable, List


def BCs_1d(
        g0: Optional[float] = 0,
        g1: Optional[float] = 0,
        bctype: Optional[Tuple[int, int]] = (0, 0),
        domain: Optional[Tuple[float, float]] = (0, 1)
) -> Tuple[Callable[[float], float], Callable[[float], float]]:
    """
    Constructs interpolation functions for 1D boundary conditions.

    This function generates two functions:
    - `u_g(x)`: Interpolates the boundary condition across the domain.
    - `u_g_prime(x)`: Provides the derivative of `u_g(x)`.

    It supports Dirichlet and Neumann boundary conditions, including mixed types.

    Parameters
    ----------
    g0 : float, optional, default=0
        Boundary value (Dirichlet) or flux (Neumann) at the left boundary (x = 0).
    g1 : float, optional, default=0
        Boundary value (Dirichlet) or flux (Neumann) at the right boundary (x = 1).
    bctype : tuple of int, optional, default=(0, 0)
        A 2-tuple specifying the type of boundary conditions:
        - (0, 0): Dirichlet at both boundaries.
        - (0, 1): Dirichlet at the left boundary, Neumann at the right.
        - (1, 0): Neumann at the left boundary, Dirichlet at the right.
    domain : tuple of float, optional, default=(0, 1)
        A 2-tuple specifying the left and right boundaries of the domain.

    Returns
    -------
    u_g : Callable[[float], float]
        A function that interpolates the boundary conditions across the domain.
    u_g_prime : Callable[[float], float]
        A function representing the derivative of `u_g(x)`.

    Raises
    ------
    ValueError
        If `bctype` is not one of (0, 0), (0, 1), or (1, 0).

    Examples
    --------
    >>> u_g, u_g_prime = BCs_1d(1, 2, bctype=(1, 0), domain=(0, 1))
    >>> u_g(0.5)
    >>> u_g_prime(0.5)
    """
    L = domain[1] - domain[0]

    if bctype == (0, 0):
        u_g = lambda x: g0 + (g1 - g0) * (x / L)
        u_g_prime = lambda x: (g1 - g0) / L
    elif bctype == (0, 1):
        u_g = lambda x: g0 + g1 * (x / L)
        u_g_prime = lambda x: g1 / L
    elif bctype == (1, 0):
        u_g = lambda x: g1 - g0 + g0 * (x / L)
        u_g_prime = lambda x: g0 / L
    else:
        raise ValueError("Only (0, 0), (0, 1), (1, 0) are valid values for the argument bctype.")
    return u_g, u_g_prime


class BCs_2d:
    """
    A class for defining and applying 2D boundary conditions in FEM.

    This class allows the user to define boundary conditions on a 2D mesh, including interpolation
    along the edges of the domain using various methods (e.g., linear, parabolic). It precomputes
    edge data for efficient evaluation of the boundary conditions.

    Attributes
    ----------
    mesh : MeshGenerator_2d
        The input mesh object.
    vertices : np.ndarray, shape=(n_vertices,)
        The (x, y) coordinates of the vertices in the domain.
    edges : np.ndarray, shape=(n_edges,)
        The edges of the domain, defined as pairs of vertex indices.
    vertex_values : np.ndarray, shape=(n_vertices,)
        Values specified at each vertex for interpolation.
    interpolations : list
        Interpolation methods for each edge. Supported methods are:
        - `linear`.
        - `quadratic`.
        - `cubic`.
        - `sine`.
        - `cosine`.
    edge_vecs : np.ndarray, shape=(n_edges, 2)
        Array of vectors along each edge, computed during initalisation.
    edge_lengths2 : np.ndarray, shape=(n_edges,)
        Array of the lengths of each edge.
    edge_functions : list
        List of callable interpolation functions for each edge.

    Methods
    -------
    construct_inhomogeneous_surface(node_coords=None)
         Constructs the inhomogeneous part of the solution `u_g(x, y)` as a callable, or if `node_coords`
         is given, as a precomputed array.
    _compute_perpendicular_distances(points)
        Computes the distances from points to all edges.
    _preocompute_edge_data()
        Precomputes edge vectors, lengths, and interpolation functions.
    """
    def __init__(
            self,
            mesh: object,
            vertex_values: Optional[np.ndarray] = None,
            interpolation_methods: Optional[list] = None,
            neumann_functions: Optional[List[Callable[[float, float], float]]] = None
    ) -> None:
        """
        Initialise the BCs_2d class to define boundary conditions on a 2D mesh.

        Parameters
        ----------
        mesh : MeshGenerator_2d
            The input mesh object. It must have the attributes `mesh`, `domain_vertices`, and
            `domain_edges`.
        vertex_values : np.ndarray, optional, default=None
            Boundary condition values at the vertices of the domain. If not specified, homogeneous
            Dirichlet boundary conditions will be assumed for all vertices.
        neumann_functions : list of callables, optional
            Neumann boundary functions g(x, y) for edges.
        interpolation_methods : list, optional, default=None
            List of interpolation methods for each edge. Supported methods are:
            - `linear`.
            - `quadratic`.
            - `cubic`.
            - `sine`.
            - `cosine`.
            If not specified, `linear' interpolation will be assumed for each edge.

        Raises
        ------
        ValueError
            If the input parameters are invalid or the mesh object is missing required attributes.
        """
        if not all(hasattr(mesh, attr) for attr in ["mesh", "domain_vertices", "domain_edges"]):
            raise ValueError("The provided mesh object is invalid. Ensure it the MeshGenerator_2d class is"
                             "correctly imported from fem_portfolio.src.mesh.mesh.")

        # Equip class with necessary attributes
        self.mesh = mesh.mesh
        self.vertices = mesh.domain_vertices
        self.edges = mesh.domain_edges

        self.vertex_values = vertex_values if vertex_values is not None else [0] * len(self.vertices)
        self.neumann_functions = neumann_functions
        self.interpolations = interpolation_methods if interpolation_methods is not None else ["linear"] * len(
            self.edges)

        self._precompute_edge_data()

    def construct_inhomogeneous_surface(
            self,
            node_coords: np.ndarray = None
    ) -> Union[Callable[[float, float], float], np.ndarray]:
        """
        Construct the surface function `u_g(x, y)` as either a callable or a precomputed array.

        Parameters
        ----------
        node_coords : ndarray, optional, default=None
            If specified, a (n_points, 2) array of coordinates for precomputeing `u_g` values on
            a structured grid.

        Returns
        -------
        u_g : Callable[[float, float], float] or np.ndarray
            A function that evaluates `u_g` at any point (x, y), or a precomputed array of `u_g` values.

        Examples
        --------
        >>> from fem_portfolio.src.mesh.mesh import MeshGenerator_2d
        >>> points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        >>> facets = [(0, 1), (1, 2), (2, 3), (3, 0)]
        >>> def refinement_func(vertices, area): return area > 0.01
        >>> mesh = MeshGenerator_2d(points, facets, refinement_func)
        >>> vertex_values = [0, 1, 0, 4]
        >>> interpolations = ["linear"] * len(mesh.domain_edges)
        >>> bcs = BCs_2d(mesh, vertex_values, interpolations)
        >>> u_g = bcs.construct_inhomogeneous_surface()
        >>> u_g(0.5, 0.5)
        """
        if node_coords is not None:
            # Precompute distances and u_g values at all nodes
            distances = self._compute_perpendicular_distances(node_coords)  # Shape: (n_nodes, n_edges)
            weights = 1 / (distances + 1e-7)  # Add small epsilon to avoid division by zero
            normalised_weights = weights / np.sum(weights, axis=1, keepdims=True)  # Shape: (n_nodes, n_edges)

            func_values = np.stack([
                func(node_coords) for func in self.edge_functions
            ], axis=1)
            u_g_array = np.sum(normalised_weights * func_values, axis=1)
            return u_g_array

        else:
            # Return callable function
            def u_g(x, y):
                # Handle scalar inputs by converting them to arrays
                is_scalar = np.isscalar(x) and np.isscalar(y)
                if is_scalar:
                    points = np.array([[x, y]])  # Shape: (1, 2)
                else:
                    # For array inputs, flatten and stack them properly
                    points = np.stack([np.atleast_1d(x).ravel(), np.atleast_1d(y).ravel()],
                                      axis=1)  # Shape: (n_points, 2)

                # Compute distances and weights
                distances = self._compute_perpendicular_distances(points)  # Shape: (1, n_edges)
                weights = 1 / (distances + 1e-7)
                normalised_weights = weights / np.sum(weights, axis=1, keepdims=True)

                # Evaluate all edge functions
                edge_values = np.array([func(points) for func in self.edge_functions])
                edge_values = edge_values.T

                # Compute the weighted sum for each point
                u_g_values = np.sum(normalised_weights * edge_values, axis=1)
                # Return scalar if input was scalar
                if is_scalar:
                    return u_g_values[0]
                else:
                    # Reshape to match the input shape
                    return u_g_values.reshape(np.atleast_1d(x).shape)
            return u_g

    def assemble_neumann_contributions(
            self,
            solver: object,
            load_vector: np.ndarray,
            boundary_edges: list
    ) -> np.ndarray:
        """
        Assemble Neumann contributions to the global load vector.

        Parameters
        ----------
        solver : BaseSolver_2d
            Solver object
        load_vector : np.ndarray
            The global load vector to be modified.
        boundary_edges : list
            List of 2-tuples representing the mesh edges that lie on the domain boundary.

        Returns
        -------
        load_vector : np.ndarray
            Modified global load vector including Neumann contributions.
        """
        for edge_idx, edge in enumerate(self.edges):
            if self.neumann_functions == None:
                break

            epsilon = getattr(solver, "epsilon", 1.0)
            g_func = self.neumann_functions[edge_idx]
            if g_func is None:
                continue

            # Get edge endpoints
            v1, v2 = self.vertices[edge[0]], self.vertices[edge[1]]
            edge_length = np.sqrt(self.edge_lengths2)[edge_idx]

            # Parametrise the edge
            x_t = lambda t: v1[0] + t * (v2[0] - v1[0])
            y_t = lambda t: v1[1] + t * (v2[1] - v1[1])

            # Integrand for load vector contributions
            def integrand(t, node_index):
                x, y = x_t(t), y_t(t)
                flux_value = g_func(x, y)
                shape_func = 1 - t if node_index == 0 else t
                return epsilon * flux_value * shape_func * edge_length

            # Integrate
            for local_idx, global_idx in enumerate(edge):
                load_vector[global_idx] += quad(integrand, 0, 1, args=(local_idx,))[0]

        return load_vector

    def _parse_function_interpolation(self, func_str):
        try:
            return eval(f"lambda g1, g2, t: {func_str}", {"np": np,
                                                     "sin": np.sin,
                                                     "cos": np.cos,
                                                     "pi": np.pi,
                                                     "e":np.e})
        except Exception as e:
            print(f"Invalid function format: {e}")
            exit(1)

    def _precompute_edge_data(self) -> None:
        """
        Precompute edge vectors, lengths, and interpolation functions.

        This method calculates the following:
        - Vectors along each edge.
        - Lengths of each edge.
        - Callable interpolation functions for each edge based on the specified methods.

        Adds the following attributes to the class instance:
        - edge_vecs : np.ndarray, shape=(n_edges, 2)
            Array containing a vector pointing along each edge.
        - edge_lengths2 : np.ndarray, shape=(n_edges,)
            Array with the squared length of each edge.
        - edge_functions : list
            List of callable interpolation functions for each edge.
        """
        # Compute edge vectors and lengths simultaneously for all edges
        edge_start = self.vertices[self.edges[:, 0]]
        edge_end = self.vertices[self.edges[:, 1]]
        self.edge_vecs = edge_end - edge_start
        self.edge_lengths2 = np.sum(self.edge_vecs ** 2, axis=1)  # Squared lengths, shape (n_edges,)

        # Precompute interpolation functions for each edge
        self.edge_functions = []
        for i, edge_vec in enumerate(self.edge_vecs):
            start = edge_start[i]
            g1, g2 = self.vertex_values[self.edges[i, 0]], self.vertex_values[self.edges[i, 1]]
            length2 = self.edge_lengths2[i]

            def make_t_function(start, vec, length2):
                """Factory function to create a parametric t_function for an edge."""
                def t_function(points):
                    points = np.atleast_2d(points)  # Ensure points have shape (n_points, 2)
                    point_vecs = points - start     # Vector from start of the edge to each point
                    t = np.einsum("ij,j->i", point_vecs, vec) / length2  # Parametric t
                    return np.clip(t, 0.0, 1.0)

                return t_function

            t_function = make_t_function(start, edge_vec, length2)

            def make_interpolation_function(g1, g2, t_function, interpolation_method):
                """Create an interpolation function based on the specified method."""
                if interpolation_method == "linear":
                    def interpolation(points, t_func=t_function):
                        t = t_func(points)
                        return g1 + t * (g2 - g1)
                elif interpolation_method == "quadratic":
                    def interpolation(points, t_func=t_function):
                        t = t_func(points)
                        return g1 + t ** 2 * (g2 - g1)
                elif interpolation_method == "cubic":
                    def interpolation(points, t_func=t_function):
                        t = t_func(points)
                        return g1 + t ** 3 * (g2 - g1)
                elif interpolation_method == "sine":
                    def interpolation(points, t_func=t_function):
                        t = t_func(points)
                        return np.sin(2 * np.pi * t) + (g2 - g1) * t + g1
                elif interpolation_method == "cosine":
                    def interpolation(points, t_func=t_function):
                        t = t_func(points)
                        return np.cos(2 * np.pi * t) + (g2 - g1) * t + g1 - 1
                else:
                    try:
                        user_func =  self._parse_function_interpolation(interpolation_method)

                        def interpolation(points, t_func=t_function):
                            t = t_func(points)
                            return user_func(g1, g2, t)
                    except:
                        raise ValueError(f"Unsupported interpolation method: {interpolation_method}")

                return interpolation

            interpolation = make_interpolation_function(g1, g2, t_function, self.interpolations[i])
            self.edge_functions.append(interpolation)

    def _compute_perpendicular_distances(self, points: np.ndarray) -> np.ndarray:
        """
        Compute the perpendicular distances from points to all edges.

        Parameters
        ----------
        points : np.ndarray, shape=(n_points, 2)
            (x, y) oordinates of the points for which distances are computed.

        Returns
        -------
        distances : ndarray, shape=(n_points, n_edges)
            Perpendicular distances to all edges.
        """
        points = np.atleast_2d(points)                          # Ensure points is (n_points, 2)
        p1_coords = self.vertices[self.edges[:, 0]]             # Start points of edges (n_edges, 2)
        edge_vecs = self.edge_vecs                              # Edge vectors (n_edges, 2)

        # Compute vector from edge start points to all input points
        point_vecs = points[:, None, :] - p1_coords[None, :, :] # Shape: (n_points, n_edges, 2)

        # Compute the parametric location t for each point along each edge
        t = np.einsum(                                          # Shape: (n_points, n_edges)
            "ijk,jk->ij",
            point_vecs, edge_vecs
        ) / self.edge_lengths2
        t_clipped = np.clip(t, 0.0, 1.0)            # Clamp t to [0, 1]

        # Compute the projection points
        projections = (p1_coords[None, :, :] +
                       t_clipped[:, :, None] * edge_vecs[None, :, :])  # Shape: (n_points, n_edges, 2)

        # Compute distances from points to their projections
        distances = np.linalg.norm(projections - points[:, None, :], axis=2)  # Shape: (n_points, n_edges)

        return distances
