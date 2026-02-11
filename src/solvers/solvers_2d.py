import logging
import numpy as np
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Union
from ..assembly.triangle_iterator_2d import TriangleIntegrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class BaseSolver_2d(ABC):
    """
    Abstract base class for 2D FEM solvers.

    This class provides common functionality for assembling FEM systems, applying boundary conditions,
    solving PDEs on 2D domains, and visualising results.

    Attributes
    ----------
    mesh : MeshGenerator_2d
        Mesh object that defines the domain's geometry and topology.
    f : Callable[[Union[float, np.ndarray], Union[float, np.ndarray]], Union[float, np.ndarray]]
        Source function for the PDE. Accepts both scalar and array inputs and returns and object of
        matching dimensions.
    bilinear_form : Callable[]
        Callable defining the weak bilinear form for the PDE.
    linear_form : Callable
        Callable defining the weak linear form for the PDE.
    bcs : BCs_2d or None
        Boundary conditions object for the domain.
    stiffness_matrix : np.ndarray
        Full stiffness matrix for the FEM system.
    load_vector : np.ndarray
        Full load vector for the FEM system.
    global_stiffness_matrix : np.ndarray
        Global stiffness matrix for the FEM system.
    global_load_vector : np.ndarray
        Global load vector for the FEM system.
    u_h : Callable
        Homogeneous part of the solution of the FEM system (i.e., excluding boundary conditions).
    u_g : Callable[[float, float], float]
        Inhomogeneous part of the solution of the FEM system. Only computed if 'bcs' is None.
    u_g_array : np.ndarray
        An array representing 'u_g(x, y)' evaluated at every node. Only computed if 'bcs' is not None.
    gradients_u_g : np.ndarray
        An array representing the values of the gradient of 'u_g(x, y)' evaluated at every node.
    u_total : Callable
        Full solution of the FEM system.
    u_total_array : np.ndarray
        An array representing the full solution to the FEM system evaluated at every node.

    Methods
    -------
    precompute_gradients()
        Precomputes basis function gradients for all elements in the mesh.
    assemble_system()
        Assembles global stiffness matrix and load vector for the FEM system.
    assemble_element_contributions(element)
        Computes the local stiffness matrix and load vector for a given element.
    solve()
        Solves the FEM system and returns the solution as a callable function. It also
        stores 'u_total' and 'u_total_array' as attributes.
    compute_pointwise_residual()

    plot(plot_type, **kwargs)

    compute_pointwise_residual
    """
    def __init__(self,
                 mesh: object,
                 f: Union[Callable[[float, float], float], Callable[[np.ndarray, np.ndarray], np.ndarray]],
                 bilinear_form: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float],
                 linear_form: Callable[[float, float], float],
                 bcs: object = None):
        """
        Initialise the 2D FEM solver.

        Parameters
        ----------
        mesh : MeshGenerator_2d
            MeshGenerator2D object from fem_portfolio.src.mesh.mesh.
        f : Callable[[float, float], float]
            Source function 'f(x, y)' for the PDE. It must accept both scalar (float) and array (np.ndarray)
            inputs and return an object of the same shape as the input. The function is called during the
            assembly of the load vector and the computation of the residual.
        bilinear_form : Callable[[ndarray, ndarray], float]
            Weak bilinear form for the PDE, involving basis function gradients and/or values.
        linear_form : Callable[[float, float], float]
            Weak linear form for the PDE< involving basis functions and the source term.
        bcs : BCs_2d, optional, default=None
            Boundary condition object for the domain.
        """
        self.mesh = mesh
        self.f = f
        self.bilinear_form = bilinear_form
        self.linear_form = linear_form
        self.bcs = bcs

        # Precompute boundary conditions if provided
        if bcs is not None:
            self.u_g_array = bcs.construct_inhomogeneous_surface(node_coords=self.mesh.node_coords)
        else:
            self.u_g = lambda x, y: 0   # Homogeneous BCs as default

        # Mesh-related attributes
        self.node_coords = mesh.node_coords
        self.lnods = mesh.lnods
        self.n_nodes = mesh.n_nodes
        self.n_elements = mesh.n_elements
        self.node_to_coords = mesh.node_to_coords
        self.coords_to_node = mesh.coords_to_node

        # Initialise system matrices and vectors
        self.stiffness_matrix = np.zeros((self.n_nodes, self.n_nodes))
        self.load_vector = np.zeros(self.n_nodes)
        self.u_h = None
        self.u_total = None

        # Precompute basis gradients for all elements
        self.precompute_gradients()

    def get_lnods(self) -> np.ndarray:
        """
        Retrieve the element connectivity array (LNODS).

        Returns
        -------
        np.ndarray
            Connectivity array of the mesh.
        """
        return self.lnods

    def get_node_coords(self) -> np.ndarray:
        """
        Retrieve the nodal coordinates of the mesh.

        Returns
        -------
        np.ndarray
            Array of nodal coordinates.
        """
        return self.node_coords

    def get_rhs_norm(self) -> float:
        """
        Retrieve the L2-norm of the RHS f(x, y).

        Returns
        -------
        float
            Value of the L2-norm.
        """
        return self.rhs_norm

    def get_residual_norm(self) -> float:
        """
        Retrieve the L2-norm of the residual.

        Returns
        -------
        float
            Value of the L2-norm.
        """
        return self._compute_fem_residual()

    def get_pointwise_residual(self) -> np.ndarray:
        """
        Retreive the pointwise residual as a NumPy array.

        Returns
        -------
        np.ndarray
            An array of residual values at a grid of points.
        """
        return self.residual_grid


    def assemble_system(self) -> None:
        """
        Assemble the global stiffness matrix and load vector for the FEM system.

        This method iterates over all elements in the mesh, computes their local contributions,
        and assembles them into the global system. It also modifies the system to handle
        Dirichlet boundary conditions.
        """
        logging.info(f"Assembling linear system, looping over the {len(self.lnods)} elements...")
        for element_idx, element in enumerate(tqdm(self.lnods)):
            element_stiffness, element_load = self.assemble_element_contributions(element)
            self._assemble_global_system(element, element_stiffness, element_load)


        # Handle Dirichlet BCs by modifying the global matrix and load vector
        all_edges = [tuple(sorted((element[i], element[(i + 1) % 3]))) for i in range(3) for element in self.lnods]
        edge_counts = Counter(map(tuple, map(sorted, all_edges)))
        boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
        boundary_nodes = set(np.ravel(boundary_edges))
        self.boundary_nodes = np.array(sorted(list(boundary_nodes)))

        # Add Neumann boundary conditions
        if self.bcs.neumann_functions is not None:
            self.load_vector = self.bcs.assemble_neumann_contributions(self, self.load_vector, boundary_edges)

        self.global_stiffness_matrix = np.delete(np.delete(self.stiffness_matrix, self.boundary_nodes, axis=0),
                                                 self.boundary_nodes, axis=1)
        self.global_load_vector = np.delete(self.load_vector, self.boundary_nodes)

    def assemble_element_contributions(self, element) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the local stiffness matrix and load vector for a single element.

        Parameters
        ----------
        element : list
            Indices of the nodes in the element.

        Returns
        -------
        stiffness_matrix : np.ndarray, shape=(3, 3)
            Local stiffness matrix for the element.
        load_vector : np.ndarray, shape=(3,)
            Local load vector for the element.
        """
        stiffness_matrix = np.zeros((3, 3))
        load_vector = np.zeros(3)

        # Get coordinates of the element's vertices
        points = np.array([self.node_coords[i] for i in element])

        # Basis functions and gradients
        eta = TriangleIntegrator.eta(*points)
        xi = TriangleIntegrator.xi(*points)
        basis_functions = [
            lambda x, y: 1 - eta(x, y) - xi(x, y),
            lambda x, y: xi(x, y),
            lambda x, y: eta(x, y)
        ]
        basis_gradients = TriangleIntegrator.compute_basis_gradients(points)

        # Boundary contribution
        u_g_values = self.u_g_array[element]
        u_g_func = lambda x, y: sum(
            self.u_g_array[element[k]] * basis_functions[k](x, y)
            for k in range(3)
        )
        grad_u_g = np.sum([
            u_g_values[k] * basis_gradients[k] for k in range(3)
        ], axis=0)

        # Stiffness matrix assembly
        for i in range(3):
            for j in range(3):
                integrand = lambda x, y: self.bilinear_form(
                    basis_gradients[j],
                    basis_gradients[i],
                    basis_functions[j](x, y),
                    basis_functions[i](x, y)
                )
                stiffness_matrix[i, j] = TriangleIntegrator.integrate_over_triangle(points, integrand)

        # Load vector assembly
        for i in range(3):
            integrand = lambda x, y: (
                    self.linear_form(basis_functions[i](x, y), self.f(x, y)) -
                    self.bilinear_form(
                        grad_u_g,
                        basis_gradients[i],
                        u_g_func(x, y),
                        basis_functions[i](x, y)
                    )
            )
            load_vector[i] = TriangleIntegrator.integrate_over_triangle(points, integrand)

        return stiffness_matrix, load_vector

    def precompute_gradients(self):
        self.gradients_u_g = []
        for element in self.mesh.lnods:
            points = np.array([self.mesh.node_coords[node] for node in element])
            grads = TriangleIntegrator.compute_basis_gradients(points)
            grad_u_g = sum(
                self.u_g_array[node] * grads[k] for k, node in enumerate(element)
            )
            self.gradients_u_g.append(grad_u_g)

        self.gradients_u_g = np.array(self.gradients_u_g)

    def _assemble_global_system(self, element, element_stiffness, element_load) -> np.ndarray:
        """

        """
        for local_i, global_i in enumerate(element):
            self.load_vector[global_i] += element_load[local_i]
            for local_j, global_j in enumerate(element):
                self.stiffness_matrix[global_i, global_j] += element_stiffness[local_i, local_j]

    def solve(self) -> Callable[[float, float], float]:
        """
        Solves the FEM system and returns the solution as a callable.
        Also precomputes solution values at nodes for efficiency.
        """
        if self.stiffness_matrix is None or self.load_vector is None:
            raise ValueError("System not assembled. Call assemble_system() first.")

        # Solve the reduced system
        U_internal = np.linalg.solve(self.global_stiffness_matrix, self.global_load_vector)

        # Reconstruct the full solution at nodes
        self.u_h_array = np.zeros(self.n_nodes)
        self.u_h_array[np.setdiff1d(range(self.n_nodes), self.boundary_nodes)] = U_internal
        self.u_total_array = self.u_h_array + self.u_g_array

        # Create callable for u_h
        from scipy.interpolate import LinearNDInterpolator
        #interpolator = LinearNDInterpolator(self.node_coords, self.u_h_array)
        #self.u_h = lambda x, y: interpolator(x, y)

        interpolator2 = LinearNDInterpolator(self.node_coords, self.u_total_array)
        self.u_total = lambda x, y: interpolator2(x, y)

        # Compute relevant quantities
        self._compute_pointwise_residual()  # Adds attribute 'self.residual_grid'
        self._compute_residual_norm()       # Adds attribute 'self.residual_norm'
        self._compute_rhs_norm()            # Adds attribute 'self.rhs_norm'

        return self.u_total, self.u_total_array

    def _compute_pointwise_residual(self) -> None:
        """
        Compute the pointwise residual R(x, y) = -epsilon * ∇²u + beta · ∇u + alpha * u - f(x, y)
        on a structured grid using NumPy.
        """
        x_grid, y_grid = self.mesh.x_grid, self.mesh.y_grid
        points_to_elements = self.mesh.points_to_elements
        resolution = self.mesh.resolution

        #Initialise residual grid
        residual_grid = np.zeros_like(x_grid)

        # Iterate over grid points
        for i in range(resolution):
            for j in range(resolution):
                x_point, y_point = x_grid[i, j], y_grid[i, j]
                element_idx = points_to_elements[i, j]

                # Find triangle containing the point
                if element_idx == -1:
                    residual_grid[i, j] = np.nan
                    continue

                # Get triangle data
                points = np.array([self.node_coords[node] for node in self.lnods[element_idx]])
                basis_gradients = TriangleIntegrator.compute_basis_gradients(points)

                grad_u_h = sum(self.u_h_array[self.lnods[element_idx][k]] * basis_gradients[k] for k in range(3))
                grad_u_g = self.gradients_u_g[element_idx]

                u = self.u_total(x_point, y_point)
                grad_u = grad_u_h + grad_u_g

                f_value = self.f(x_point, y_point)

                epsilon = getattr(self, "epsilon", 1.0)
                beta = getattr(self, "beta", np.array([0.0, 0.0]))
                alpha = getattr(self, "alpha", 0.0)

                convection = np.dot(beta, grad_u) if hasattr(self, "beta") else 0
                reaction = alpha * u if hasattr(self, "alpha") else 0
                diffusion = - epsilon * laplace(u)
                residual_grid[i, j] = diffusion + convection + reaction - f_value

        self.residual_grid = residual_grid

    def _compute_fem_residual(self) -> None:
        """
        Compute the FEM residual as:

            R = A u - F

        where:
        - A is the global stiffness matrix.
        - u is the computed FEM solution.
        - F is the load vector.

        This is the most FEM-consistent way to compute residuals.
        """

        if self.stiffness_matrix is None or self.load_vector is None:
            raise ValueError("System not assembled. Call assemble_system() first.")

        # Compute FEM residual
        residual_vector = self.stiffness_matrix @ self.u_total_array - self.load_vector

        # Compute residual norm
        self.residual_norm = np.linalg.norm(residual_vector, ord=2)  # L2 norm

        print(f"Computed FEM Residual Norm: {self.residual_norm}")

    def _compute_residual_norm(self) -> None:
        """

        """
        x_grid, y_grid = self.mesh.x_grid, self.mesh.y_grid
        residual_grid = getattr(self, "residual_grid", None)
        if residual_grid is None:
            print("The method 'self._compute_pointwise residual()' has to be called first.")
        else:
            # Compute the area of each grid cell
            dx = x_grid[0, 1] - x_grid[0, 0]
            dy = y_grid[1, 0] - y_grid[0, 0]
            cell_area = dx * dy

            # Compute the L2 norm of the residual
            self.residual_norm = np.sqrt(np.nansum(residual_grid ** 2 * cell_area))

    def _compute_rhs_norm(self) -> None:
        """
        Compute the L2-norm of the RHS of the PDE f(x, y).

        Returns
        -------
        l2_norm : float
            Computed value of the L2-norm of f(x, y).
        """
        x_grid, y_grid = self.mesh.x_grid, self.mesh.y_grid
        points_to_elements = self.mesh.points_to_elements
        resolution = self.mesh.resolution

        # Evaluate the solution callable on the grid
        f_grid = self.f(x_grid, y_grid)

        # Remove values outside the domain
        for i in range(resolution):
            for j in range(resolution):
                element = points_to_elements[i, j]
                if element == -1:
                    f_grid[i, j] = np.nan
                    continue

        # Compute the area of each grid cell (assuming uniform grid spacing)
        dx = x_grid[0, 1] - x_grid[0, 0]
        dy = y_grid[1, 0] - y_grid[0, 0]
        cell_area = dx * dy

        # Compute the L2 norm of f
        self.rhs_norm = np.sqrt(np.nansum(f_grid ** 2 * cell_area))

    def plot(
            self,
            data_type: str,
            plot_style: str,
            filled: Optional[bool] = True,
            show: Optional[bool] = True,
            plot_mesh: Optional[bool] = True
    ) -> None:
        """
        Plotting method for FEM results.

        Parameters
        ----------
        data_type : str
            Type of data to plot. Options:
            - 'solution': Plot the FEM solution 'u(x, y)'.
            - 'residual': Plot the pointwise residual.
            - 'source' : Plot the source term 'f(x, y)'.
        plot_style : str
            Style of the plot. Options:
            - 'contour': 2D contour plot.
            - '3d': 3D surface plot.
            - 'scatter': Node-based scatter plot.
        filled : bool, optional, default=True
            If True, use filled contours. Applicable only for 'contour' plots.
        show : bool, optional, default=True
            If True, display the plot immediately.
        plot_mesh : bool, optional, default=True
            If True, overlay the mesh on the plot (applicable to contour plots for the solution).

        Raises
        ------
        ValueError
            If an unsupported data type or plot style is specified.

        Examples
        --------
        >>> solver.plot("solution", "contour", filled=True)
        >>> solver.plot(data_type="residual", plot_style="3d")
        >>> solver.plot(data_type="source", plot_style="contour", filled=False)
        """
        # Plotting settings
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")
        plt.rcParams.update({"font.size": 20})

        x_grid, y_grid = self.mesh.x_grid, self.mesh.y_grid

        # Determine the data to plot
        if data_type == "solution":
            z_values = self.u_total(x_grid, y_grid) if plot_style != "scatter" else self.u_total_array
            title = "FEM interpolated solution" if plot_style != "scatter" else "FEM solution at mesh nodes"
        elif data_type == "residual":
            z_values = self.residual_grid if plot_style != "scatter" else self.residual_grid.flatten()
            title = "Residual"
        elif data_type == "source":
            z_values = self.f(x_grid, y_grid) if plot_style != "scatter" else self.f(self.node_coords[:, 0],
                                                                                     self.node_coords[:, 1])
            title = r"$f(x, y)$"
        else:
            raise ValueError(f"Unsupported data type '{data_type}'. Use 'solution', 'residual', or 'source'.")

        # Choose the plotting style
        if plot_style == "contour":
            self._plot_contour(x_grid, y_grid, z_values, title, filled, show, plot_mesh)
        elif plot_style == "3d":
            self._plot_3d(x_grid, y_grid, z_values, title, show, plot_mesh)
        elif plot_style == "scatter":
            self._plot_scatter_nodes(z_values, title, show, plot_mesh)
        else:
            raise ValueError(f"Unsupported plot style '{plot_style}'. Use 'contour', '3d', or 'scatter'.")

    def _plot_contour(
            self,
            x_grid: np.ndarray,
            y_grid: np.ndarray,
            z_values: np.ndarray,
            title: str,
            filled: bool,
            show: bool,
            plot_mesh: bool
    ) -> None:
        """
        Helper method for 2D contour plots.

        Parameters
        ----------
        x_grid, y_grid : np.ndarray
            2D arrays representing the grid coordinates.
        z_values : np.ndarray
            2D array of values to plot.
        title : str
            Title of the plot.
        filled : bool
            If True, use filled contours. If False, use line contours.
        show : bool
            If True, display the plot immediately.
        plot_mesh : bool
            If True, overlay the mesh structure on the plot.
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Mask values outside the domain
        points_to_elements = self.mesh.points_to_elements
        z_values = np.where(points_to_elements != -1, z_values, np.nan)
        if filled:
            c = ax.contourf(x_grid, y_grid, z_values, cmap="magma", levels=50)
        else:
            c = ax.contour(x_grid, y_grid, z_values, cmap="magma", levels=50)
        fig.colorbar(c, ax=ax, label=title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)

        if plot_mesh:
            for element in self.mesh.lnods:
                element_coords = self.node_coords[element]
                for i in range(3):
                    x_start, y_start = element_coords[i]
                    x_end, y_end = element_coords[(i + 1) % 3]
                    ax.plot([x_start, x_end], [y_start, y_end], color="k", linewidth=0.5, alpha=0.4)

        if show:
            plt.show()

    def _plot_3d(
            self,
            x_grid: np.ndarray,
            y_grid: np.ndarray,
            z_values: np.ndarray,
            title: str,
            show: bool,
            plot_mesh: bool
    ) -> None:
        """
        Helper method for 3D surface plots.

        Parameters
        ----------
        x_grid, y_grid : np.ndarray
            2D arrays representing the grid coordinates.
        z_values : np.ndarray
            2D array of values to plot.
        title : str
            Title of the plot.
        show : bool
            If True, display the plot immediately.
        plot_mesh : bool
            If True, overlay the mesh structure on the plot.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Mask values outside the domain
        points_to_elements = self.mesh.points_to_elements
        z_values = np.where(points_to_elements != -1, z_values, np.nan)

        # Plot the 3D surface
        surface = ax.plot_surface(x_grid, y_grid, z_values, cmap="magma", edgecolor="none", alpha=0.9)
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10, label=title)

        # Overlay the triangular mesh in 3D
        if plot_mesh:
            for element in self.mesh.lnods:
                element_coords = self.node_coords[element]
                for i in range(3):
                    start_node = None
                    x_start, y_start = element_coords[i]
                    x_end, y_end = element_coords[(i + 1) % 3]
                    start_idx = self.coords_to_node[tuple(element_coords[i])]
                    end_idx = self.coords_to_node[tuple(element_coords[(i + 1) % 3])]
                    z_start, z_end = self.u_total_array[start_idx], self.u_total_array[end_idx]
                    ax.plot([x_start, x_end], [y_start, y_end], [z_start, z_end], color="k", linewidth=0.5, alpha=0.9)

        # Add contour at z=0
        ax.contour(x_grid, y_grid, z_values, zdir="z", offset=0, cmap="magma", levels=50)

        # Set labels and title
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel(title)
        ax.set_title(f"{title} (3D with Mesh and Level Lines)")

        # Adjust view for better visualization
        ax.view_init(elev=25, azim=-45)

        if show:
            plt.show()

    def _plot_scatter_nodes(
            self,
            z_values: np.ndarray,
            title: str,
            show: bool,
            plot_mesh: bool
    ) -> None:
        """
        Helper method for node-based scatter plots.

        Parameters
        ----------
        z_values : ndarray
            1D array of values at the nodes.
        title : str
            Title of the plot.
        show : bool
            If True, display the plot immediately.
        plot_mesh : bool
            If True, overlay the mesh structure on the plot.
        """
        x_coords = self.node_coords[:, 0]
        y_coords = self.node_coords[:, 1]

        plt.figure(figsize=(10, 8))
        plt.scatter(x_coords, y_coords, c=z_values, cmap="magma", s=20)
        plt.colorbar(label=title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(title)

        if plot_mesh:
            for element in self.mesh.lnods:
                element_coords = self.node_coords[element]
                for i in range(3):
                    x_start, y_start = element_coords[i]
                    x_end, y_end = element_coords[(i + 1) % 3]
                    plt.plot([x_start, x_end], [y_start, y_end], color="k", linewidth=0.5, alpha=0.4)

        if show:
            plt.show()


class Poisson_2d(BaseSolver_2d):
    def __init__(self, mesh, f, bcs):
        bilinear_form = lambda grad_i, grad_j, phi_i, phi_j: np.dot(grad_i, grad_j)
        linear_form = lambda phi, f: phi * f
        super().__init__(mesh, f, bilinear_form, linear_form, bcs)


class ConvectionDiffusion_2d(BaseSolver_2d):
    def __init__(self, mesh, f, epsilon, beta, alpha, bcs):
        self.epsilon = epsilon
        self.beta = beta
        self.alpha = alpha
        bilinear_form = lambda grad_i, grad_j, phi_i, phi_j: (
            epsilon * np.dot(grad_i, grad_j) +
            np.dot(beta, grad_i) * phi_j +
            alpha * phi_i * phi_j
        )
        linear_form = lambda phi, f: phi * f
        super().__init__(mesh, f, bilinear_form, linear_form, bcs)




