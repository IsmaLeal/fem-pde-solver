import numpy as np
from .mesh.mesh import MeshGenerator_2d
from .BCs.boundary_conditions import BCs_2d
from .solvers.solvers_2d import ConvectionDiffusion_2d, Poisson_2d


def solve_fem(domain='square', pde='poisson', resolution=50, epsilon=0.8, beta=(0.0, 0.0), alpha=0.0,
              max_triangle_area=0.01, f=None, points=None, facets=None, vertex_values=None, interpolations=None):
    """
    Solves a PDE using FEM based on given parameters.

    Returns:
        solver (Poisson_2d or ConvectionDiffusion_2d object)
    """

    # Define source function
    if f is None:
        f = lambda x, y: np.zeros_like(x, dtype=float)

    # Define domain geometry
    if domain == 'star' and points is None:
        num_points = 30
        points = [
            ((0.5 + 0.2 * np.sin(5 * 2 * np.pi * i / num_points)) * np.cos(2 * np.pi * i / num_points),
             (0.5 + 0.2 * np.sin(5 * 2 * np.pi * i / num_points)) * np.sin(2 * np.pi * i / num_points))
            for i in range(num_points)
        ]
        facets = [(i, (i + 1) % num_points) for i in range(num_points)]
        vertex_values = [0 for _ in range(len(points))]
        interpolations = ['linear' for _ in range(len(facets))]
    elif domain == 'square' and points is None:
        points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        facets = [(0, 1), (1, 2), (2, 3), (3, 0)]
        vertex_values = [0, 1, 1, 0]
        interpolations = ['linear', 'linear', 'linear', 'linear']
    elif domain == 'pentagon' and points is None:
        points = [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 1.5)]
        facets = [(0, 1), (1, 2), (2, 4), (4, 3), (3, 0)]
        vertex_values = [1, 0.5, -2, 2, 0.5]
        interpolations = ['cosine', 'linear', 'quadratic', 'sine', 'linear']
    elif points is None or facets is None:
        raise ValueError('For a custom domain, "points" and "facets" must be provided.')

    # Mesh generation
    mesh = MeshGenerator_2d(points, facets, lambda v, a: a > max_triangle_area, resolution)

    if vertex_values is None:
        vertex_values = [0] * len(points)
    if interpolations is None:
        interpolations = ['linear'] * len(facets)
    bcs = BCs_2d(mesh, vertex_values, interpolations)

    # Select and solve PDE
    if pde == 'poisson':
        solver = Poisson_2d(mesh, f, bcs)
    elif pde == 'convection_diffusion':
        solver = ConvectionDiffusion_2d(mesh, f, epsilon, beta, alpha, bcs)

    solver.assemble_system()
    solver.solve()
    return solver
