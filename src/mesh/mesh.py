import numpy as np
from tqdm import tqdm
from meshpy import triangle
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Callable, Union

def generate_uniform_mesh(
        N: int,
        domain: Optional[Tuple[float, float]] = (0, 1)
) -> np.ndarray:
    """
    Generates uniform mesh.

    Parameters
    ----------
    N : int
        Number of elements. There will be N+1 nodes in the mesh. Must be greater than 0.
    domain : tuple of float, optional, default=(0, 1)
        A 2-tuple specifying the boundary limits. The left boundary must be 0.

    Returns
    -------
    x_nodes : np.ndarray
        Array of uniformly spaced nodes within the specified domain.
    """
    if N <= 0:
        raise ValueError("N should be a positive integer.")

    if (domain[1] <= domain[0]) | (domain[0] != 0):
        raise ValueError("Specify a valid domain. Left boundary must be 0, and right boundary has to be greater than 0.")

    x_nodes = np.linspace(domain[0], domain[1], N+1)
    return x_nodes


def generate_nonuniform_mesh(
        N: int,
        clustering_factor: Optional[float] = 2,
        domain: Optional[Tuple[float, float]] = (0, 1)
) -> np.ndarray:
    """
    Generate non-uniform mesh with polynomial clustering.

    Parameters
    ----------
    N : int
        Number of elements. There will be N+1 nodes.
    clustering_factor : float, optional
        The order of the polynomial transformation that will cluster the nodes towards
        the right boundary. A value of 1 leaves the nodes uniformly spaced. Default is 2.
    domain : tuple, optional
        A 2-tuple specifying the boundary limits. The left boundary needs to be 0.
        Default is (0, 1).

    Returns
    -------
    x_nodes : np.ndarray
        Array of the mesh of nodes.
    """
    # Compute domain length
    L = domain[1] - domain[0]

    # Generate N+1 uniformly spaced nodes between 0 and 1
    s = np.linspace(0, 1, N+1)

    # Apply transformation
    x_nodes =  (1 - (1 - s) ** clustering_factor) * L

    return x_nodes


def refine_mesh_1d(
        x_nodes: np.ndarray,
        errors: np.ndarray,
        tol: float
) -> np.ndarray:
    """
    Refines the mesh based on error indicators.

    Parameters
    ----------
    x_nodes : ndarray
        Current 1D mesh nodes.
    errors : ndarray
        Error indicators for each element.
    tol : float
        Refinement tolerance.

    Returns
    -------
    refined_nodes : ndarray
        Refined 1D mesh nodes.
    """
    refine_indices = np.where(errors > tol)[0]
    if len(refine_indices) == 0:
        return x_nodes  # No refinement needed

    new_nodes = []
    for idx in refine_indices:
        midpoint = 0.5 * (x_nodes[idx] + x_nodes[idx + 1])
        new_nodes.append(midpoint)

    return np.sort(np.concatenate((x_nodes, new_nodes)))


class MeshGenerator_2d:
    """
    A class for generating and refining 2D triangular meshes using MeshPy.

    This class enables users to:
    - Define custom domains and facets for 2D triangular meshes.
    - Apply user-defined refinement criteria to control mesh resolution.
    - Visualise the resulting mesh with Matplotlib.

    Attributes
    ----------
    domain_vertices : np.ndarray, shape=(n_vertices, 2)
        Array of (x, y) coordinates representing the vertices of the polygonal domain.
    domain_edges : np.ndarray, shape=(n_edges, 2)
        Array of vertex index pairs defining the edges of the polygonal domain.
    mesh : MeshPy.Triangle.Mesh
        A refined MeshPy triangular mesh object.
    n_nodes : int
        Total number of nodes in the mesh.
    n_elements : int
        Total number of triangular elements in the mesh.
    node_coords : np.ndarray, shape=(n_nodes, 2)
        Array of (x, y) coordinates of the nodes in the mesh.
    lnods : np.ndarray, shape=(n_elements, 3)
        Connectivity array of the mesh with shape (n_elements, 3), where each row specifies the global indices
        of the three nodes forming a triangle.
    node_to_coords : dict
        Mapping from global node indices to their (x, y) coordinates.
    coords_to_node : dict
        Mapping from (x, y) coordinates to node indices.
    resolution : int
        Number of grid points along each axis for masking external points. Defaults to 50.
    points_to_elements : ndarray

    Methods
    -------
    _map_points_to_elements(resolution)
        Maps points on a structured grid to the elements containing them.
    plot()
        Visualises the 2D mesh with Matplotlib.
    """
    def __init__(
            self,
            points: List[Tuple[float, float]],
            facets: List[Tuple[int, int]],
            refinement_func: Callable[[Tuple[object, object, object], float], bool],
            resolution: Optional[int] = 50
    ) -> None:
        """
        Initialise the MeshGenerator_2d class to create a triangular mesh.

        Parameters
        __________
        points : list of tuple
            List of (x, y) coordinates defining the vertices of the polygonal domain.
        facets : list of tuple
            List of 2-tuples representing indices of vertices connected by domain edges.
        refinement_func : Callable[[Tuple[np.ndarray, float]], bool]
            A user-defined function that determines whether to refine a triangle based
            on its vertices and area.
        resolution : int, optional, default=50
            Number of grid points along each axis for structured grid generation.

        Raises
        ------
        ValueError
            If the input points or facets are invalid.

        Examples
        --------
        >>> points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        >>> facets = [(0, 1), (1, 2), (2, 3), (3, 0)]
        >>> def refinement_func(vertices, area): return area > 0.01
        >>> mesh = MeshGenerator_2d(points, facets, refinement_func)
        """
        self.domain_vertices = np.array(points)
        self.domain_edges = np.array(facets)

        # Create MeshPy mesh object and add as attribute to the class
        mesh_info = triangle.MeshInfo()
        mesh_info.set_points(self.domain_vertices)
        mesh_info.set_facets(self.domain_edges)
        self.mesh = triangle.build(mesh_info, refinement_func=refinement_func)

        # Add useful objects as attributes to the class
        self.node_coords = np.array(self.mesh.points)
        self.lnods = np.array(self.mesh.elements)
        self.n_nodes = len(self.node_coords)
        self.n_elements = len(self.lnods)
        self.node_to_coords = {i: coords for i, coords in enumerate(self.node_coords)}
        self.coords_to_node = {tuple(coords): i for i, coords in enumerate(self.node_coords)}

        # Get mask of external points
        self.resolution = resolution
        self.points_to_elements = self._map_points_to_elements(resolution)

    def plot(self) -> None:
        """
        Visualise the 2D mesh using Matplotlib.

        This method plots the triangular elements with:
        - Black outlines for the triangular elements.
        - Red scatter points for the mesh nodes.

        Notes
        -----
        - Ensure `plt.show()` is called if the plot does not render in your environment.

        Examples
        --------
        >>> points = [(0, 0), (1, 0), (0, 1)]
        >>> facets = [(0, 1), (1, 2), (2, 0)]
        >>> def refinement_func(vertices, area): return area > 0.01
        >>> mesh_gen = MeshGenerator_2d(points, facets, refinement_func)
        >>> mesh_gen.plot()
        """
        for triangle_nodes in self.lnods:
            triangle_coords = [self.node_coords[i] for i in triangle_nodes]
            x, y = zip(*triangle_coords + [triangle_coords[0]])  # Close the triangle
            plt.plot(x, y, "k-")

        x_coords, y_coords = zip(*self.mesh.points)
        plt.scatter(x_coords, y_coords, s=10, c="r")
        plt.gca().set_aspect("equal")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Pentagonal domain')
        plt.show()

    def _map_points_to_elements(self, resolution: int) -> np.ndarray:
        """
        Map points on a structured grid to the elements containing them.

        Parameters
        ----------
        resolution : int
            Number of grid points along each axis for the structured grid.

        Returns
        -------
        map : np.ndarray, shape=(resolution, resolution)
            Array where each entry corresponds to the index of the triangle containing the
            grid point. Grid points outside the domain are assigned the value -1.

        Notes
        -----
        This is an internal method intended for generating a point-to-element mapping
        for structured grids. It uses barycentric coordinates to determine element
        containment.
        """
        # Extract domain bounds
        x_coords, y_coords = zip(*self.node_coords)
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Create grid for evaluating the callable solution
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        self.x_grid, self.y_grid = np.meshgrid(x, y)

        # Mask points outside the domain
        map = np.full_like(self.x_grid, fill_value=-1, dtype=np.int_)
        print(f"Mapping points of a {resolution}x{resolution} grid to their corresponding triangle...")
        for i in tqdm(range(resolution)):
            for j in range(resolution):
                result = self._find_containing_triangle((self.x_grid[i, j], self.y_grid[i, j]))
                if result != None:
                    map[i, j] = result
        return map

    def _find_containing_triangle(self, point) -> Union[int, None]:
        """
        Identify the triangle in the mesh that contains a given point.

        Parameters
        ----------
        point : tuple of float
            The (x, y) coordinates of the point to locate.

        Returns
        -------
        triangle : int or None
            The index of the element (triangle) containing the point, or None if
            the point is outside the domain.

        Notes
        -----
        The method uses barycentric coordinates to determine whether the point lies
        inside a triangle. It assumes counterclockwise vertex ordering for valid results.
        """
        x, y = point
        tolerance = 1e-9

        for element_idx, element in enumerate(self.lnods):
            # Get the coordinates of the triangle's vertices
            vertices = np.array([self.node_coords[i] for i in element])
            x1, y1 = vertices[0]
            x2, y2 = vertices[1]
            x3, y3 = vertices[2]

            # Check orientation (ensure counterclockwise order)
            area = 0.5 * ((x1 * (y2 - y3)) + (x2 * (y3 - y1)) + (x3 * (y1 - y2)))
            if area < 0:
                vertices = vertices[[0, 2, 1]]  # Reverse order to make it counterclockwise
                x1, y1 = vertices[0]
                x2, y2 = vertices[1]
                x3, y3 = vertices[2]

            # Compute barycentric coordinates
            denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
            alpha = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
            beta = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom
            gamma = 1 - alpha - beta

            # Check if the point is inside the triangle
            if -tolerance <= alpha <= 1 + tolerance and -tolerance <= beta <= 1 + tolerance and -tolerance <= gamma <= 1 + tolerance:
                return int(element_idx)  # Return the triangle as a list of node indices

        return None  # Point is outside the domain
