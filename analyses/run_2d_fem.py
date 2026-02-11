import numpy as np
import matplotlib.pyplot as plt
from src.mesh.mesh import MeshGenerator_2d
from src.BCs.boundary_conditions import BCs_2d
from src.solvers.solvers_2d import ConvectionDiffusion_2d


def f(x, y): #return np.exp(-x**2 - y ** 2)
    #return 2 * x * (1 - x) + 2 * y * (1 - y)
    return np.full_like(x, fill_value=2) #np.sin(2 * np.pi * x * np.cos(y))


def needs_refinement(vertices, area):
    return area > 0.005


def star_points(num_points, radius_func):
    points = []
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        r = radius_func(angle)
        x, y = r * np.cos(angle), r * np.sin(angle)
        points.append((x, y))
    return points

def radius_func(theta):
    return 0.5 + 0.2 * np.tan(5 * theta)


domain = "pentagon"
if domain == "star":
    num_points = 50
    points = star_points(num_points, radius_func)
    facets = [(i, (i + 1) % num_points) for i in range(num_points)]
    vertex_values = [i / 40 for i in range(len(points))]
    interpolations = ["parabolic" for i in range(len(facets))]
elif domain == "pentagon":
    points = [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 1.5)]
    facets = [(0, 1), (1, 2), (2, 4), (4, 3), (3, 0)]
    vertex_values = [1, 0.5, 0.5, 2, 0.5]
    interpolations = ["cosine", "linear", "parabolic", "sine", "linear"]
elif domain == "square":
    points = [(0, 0), (1, 0), (1, 1), (0, 1)]
    facets = [(0, 1), (1, 2), (2, 3), (3, 0)]
    vertex_values = [0, 0, 0, 0]
    interpolations = ["cosine", "linear", "linear", "linear"]
else:
    raise ValueError("'domain' must be either 'square', 'pentagon', or 'star'.")

mesh_resolution = 50
mesh = MeshGenerator_2d(points, facets, needs_refinement, mesh_resolution)
mesh.plot()
bcs = BCs_2d(mesh, vertex_values, interpolations)

#mesh.plot()

solver = ConvectionDiffusion_2d(mesh, f, 0.02, (0, 1), 0, bcs)
solver.assemble_system()
solver.solve()
solver.plot("solution", "3d", show=False)
plt.show()

