import warnings
import numpy as np
from typing import Callable, Tuple
from scipy.integrate import dblquad

class TriangleIntegrator:
    """
    A utility class for 2D FEM providing tools for integration over triangular elements and basis function evaluation.

    This class provides tools for:
    - Numerically integrating functions over triangular elements using Gaussian quadrature.
    - Defining and evaluating linear nodal basis functions (eta and xi) for triangular elements.
    - Computing the gradients of these basis functions for use in FEM assembly.

    Attributes
    ----------
    None (this class only contains static methods).

    Methods
    -------
    integrate_over_triangle(points, integrand)
        Computes the integral of a given function over a triangular element using SciPy's Gaussian quadrature.
    eta(vertices)
        Returns a lambda function representing the linear nodal basis function `eta(x, y)` for a triangle.
    xi((x1, y1), (x2, y2), (x3, y3))
        Returns a lambda function representing the linear nodal basis function `xi(x, y)` for a triangle.
    compute_basis_gradients(points)
        Computes the gradients of the basis functions for a triangular element.
    """
    @staticmethod
    def integrate_over_triangle(
            points: np.ndarray, integrand: Callable[[float, float], float]
    ) -> float:
        """
        Integrate the callable `integrand` over one triangle defined by its vertices using Gaussian quadrature.

        This method uses SciPy's `dblquad` function to perform the integration. For triangles without vertical sides,
        the method splits the triangle into two smaller sub-triangles using a vertical line through the central vertex
        (sorted by x-coordinate). Each sub-triangle is then integrated separately. If the triangle has a vertical side,
        no splitting is required as one single integral suffices.

        Parameters
        ----------
        points : ndarray, shape=(3, 2)
            Array with the (x, y) coordinates of the three vertices of the triangle. The vertices
            must be ordered in counter-clockwise direction.
        integrand : Callable[[float, float], float]
            A callable function representing the integrand to evaluate over the triangle.

        Returns
        -------
        total_integral : float
            Value of the surface integral over the triangle.

        Notes
        -----
        - Vertices must be provided in counter-clockwise order.
        - Infinite gradients resulting from vertical edges are handled by ignoring runtime warnings.
        - The sign of the integral is adjusted based on triangle orientation to ensure consistency.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.integrate import dblquad
        >>> def integrand(x, y): return x + y
        >>> points = np.array([[0, 0], [1, 0], [0, 1]])
        >>> TriangleIntegrator.integrate_over_triangle(points, integrand)
        """
        # Sort vertices by ascending x-coordinate
        x_sorted = points[np.argsort(points[:, 0])]
        (x1, y1), (x2, y2), (x3, y3) = x_sorted  # leftmost, central, rightmost points

        # Ignore warnings for infinite slopes (vertical edges)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # Define equations for the edges of the triangle
        m12 = (y2 - y1) / (x2 - x1)     # Slope of edge between (x1, y1) and (x2, y2)
        y12 = lambda x: m12 * (x - x1) + y1

        m23 = (y3 - y2) / (x3 - x2)     # Slope of edge between (x2, y2) and (x3, y3)
        y23 = lambda x: m23 * (x - x2) + y2

        m31 = (y1 - y3) / (x1 - x3)     # Slope of edge between (x3, y3) and (x1, y1)
        y31 = lambda x: m31 * (x - x3) + y3

        # Handle triangles with vertical edges
        if x1 == x2:
            total_integral, _ = dblquad(
                integrand,
                x1, x3,
                lambda x: y23(x),   # y-lower limit
                lambda x: y31(x)    # y-upper limit
            )
            if m23 < m31:           # Adjust sign if edge orientation differs
                total_integral = -1 * total_integral
        elif x2 == x3:
            total_integral, _ = dblquad(
                integrand,
                x1, x3,
                lambda x: y31(x),
                lambda x: y12(x)
            )
            if m12 < m31:
                total_integral = -1 * total_integral
        # If the triangle has no vertical sides, split the triangle
        else:
            # Left sub-triangle
            left_integral, _ = dblquad(
                integrand,
                x1, x2,
                lambda x: y31(x),
                lambda x: y12(x)
            )
            if m12 < m31:
                left_integral = -1 * left_integral

            # Right sub-triangle
            right_integral, _ = dblquad(
                integrand,
                x2, x3,
                lambda x: y31(x),
                lambda x: y23(x)
            )
            if m31 < m23:
                right_integral = -1 * right_integral

            # Total integral
            total_integral = left_integral + right_integral
        return total_integral

    @staticmethod
    def eta(*args: Tuple[float, float]) -> Callable[[float, float], float]:
        """
        Constructs an anonymous linear nodal basis function `eta(x,y)` for a triangular element
        in counter-clockwise order.

        Parameters
        ----------
        *args : tuple of float
            Three tuples representing the vertices of the triangle in counter-clockwise order:
            - (x1, y1): Coordinates of the first vertex.
            - (x2, y2): Coordinates of the second vertex.
            - (x3, y3): Coordinates of the third vertex.

        Returns
        -------
        Callable[[float, float], float]
            A lambda function representing eta(x, y), which satisfies:
            - eta(x1, y1) = eta(x2, y2) = 0.
            - eta(x3, y3) = 1.

        Raises
        ------
        ValueError
            If the number of vertices provided in `args` is not exactly three.

        Examples
        --------
        >>> eta = TriangleIntegrator.eta((0, 0), (1, 0), (0, 1))
        >>> eta(0.5, 0.5)
        """
        if len(args) != 3:
            raise ValueError("eta requires exactly 3 inputs: (x1, y1), (x2, y2), (x3, y3)")
        (x1, y1), (x2, y2), (x3, y3) = args

        if x2 != x1:
            constant = 1 / (y3 - y1 - ((x3 - x1) * (y2 - y1)) / (x2 - x1))
            return lambda y, x: constant * (y - y1 - ((y2 - y1) / (x2 - x1)) * (x - x1))
        else:
            return lambda y, x: (x - x1) / (x3 - x1)

    @staticmethod
    def xi(*args: Tuple[float, float]) -> Callable[[float, float], float]:
        """
        Constructs an anonymous linear nodal basis function `xi(x,y)` for a triangular element
        in counter-clockwise order.

        Parameters
        ----------
        *args : tuple of float
            Three tuples representing the vertices of the triangle in counter-clockwise order:
            - (x1, y1): Coordinates of the first vertex.
            - (x2, y2): Coordinates of the second vertex.
            - (x3, y3): Coordinates of the third vertex.

        Returns
        -------
        Callable[[float, float], float]
            A lambda function representing xi(x, y), which satisfies:
            - eta(x1, y1) = eta(x3, y3) = 0.
            - eta(x2, y2) = 1.

        Raises
        ------
        ValueError
            If the number of vertices provided in `args` is not exactly three.

        Examples
        --------
        >>> xi = TriangleIntegrator.xi((0, 0), (1, 0), (0, 1))
        >>> xi(0.5, 0.5)
        """
        if len(args) != 3:
            raise ValueError("eta requires exactly 6 inputs: (x1, y1), (x2, y2), (x3, y3)")
        (x1, y1), (x2, y2), (x3, y3) = args

        if x2 != x1:
            eta_func = TriangleIntegrator.eta(*args)
            return lambda y, x: (x - eta_func(y, x) * (x3 - x1) - x1) / (x2 - x1)
        else:
            return lambda y, x: (y - ((y3 - y1) * (x - x1) / (x3 - x1)) - y1) / (y2 - y1)

    @staticmethod
    def compute_basis_gradients(points: np.ndarray) -> np.ndarray:
        """
        Computes the gradients of the linear nodal basis functions for a triangular element. The
        three nodal basis functions are `1 - xi - eta`, `xi`, and `eta`.

        Parameters
        ----------
        points : ndarray, shape=(3, 2)
            The coordinates of the triangle's vertices, ordered counter-clockwise:
            - points[0]: First vertex (x1, y1)
            - points[1]: Second vertex (x2, y2)
            - points[2]: Third vertex (x3, y3)

        Returns
        -------
        gradients : np.ndarray, shape=(3, 2)
            An array containing the gradients of the three basis functions, where:
            - gradients[i, 0] is the x-derivative of the i-th basis function.
            - gradients[i, 1] is the y-derivative of the i-th basis function.
        """
        (x1, y1), (x2, y2), (x3, y3) = points
        area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

        gradients = np.array([
            [(y2 - y3), (x3 - x2)],
            [(y3 - y1), (x1 - x3)],
            [(y1 - y2), (x2 - x1)],
        ]) / (2 * area)
        return gradients