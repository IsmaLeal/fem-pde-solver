import numpy as np
import matplotlib.pyplot as plt
from src.mesh.mesh import MeshGenerator_2d
from src.BCs.boundary_conditions import BCs_2d
from src.solvers.solvers_2d import ConvectionDiffusion_2d

def f(x, y, epsilon=0.8, beta=(0.5, 2.0), alpha=0.0):
    return x**2 - 4 * np.cos(4 * np.pi * x * y)


def needs_refinement(vertices, area):
    #return area > 0.1  # Moderate refinement threshold
    # Calculate the average x-coordinate of the triangle's vertices
    avg_x = np.mean([vertex[0] for vertex in vertices])

    # Adjust refinement threshold based on proximity to the right boundary
    weight = 1.0 + 2.0 * avg_x  # Increase refinement closer to x=1
    #return area > (0.01 / weight)
    return area > 0.05


def star_points(num_points, radius_func):
    points = []
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        r = radius_func(angle)
        x, y = r * np.cos(angle), r * np.sin(angle)
        points.append((x, y))
    return points


def radius_func(theta):
    return 0.5 + 0.2 * np.sin(5 * theta)


# Define domain
domain = 'square'
if domain == 'star':
    num_points = 30
    points = star_points(num_points, radius_func)
    facets = [(i, (i + 1) % num_points) for i in range(num_points)]
    vertex_values = [0 for i in range(len(points))]
    interpolations = ['linear' for i in range(len(facets))]
    neumann_functions = None
elif domain == 'pentagon':
    points = [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 1.5)]
    facets = [(0, 1), (1, 2), (2, 4), (4, 3), (3, 0)]
    vertex_values = [1, 0.5, -2, 2, 0.5]
    interpolations = ['cosine', 'linear', 'quadratic', 'sine', 'linear']
    neumann_functions = None
elif domain == 'square':
    points = [(0, 0), (1, 0), (1, 1), (0, 1)]
    facets = [(0, 1), (1, 2), (2, 3), (3, 0)]
    vertex_values = [0, 1, -1, 0]
    interpolations = ['linear', 'quadratic', 'linear', 'linear']


    def g_bottom(x, y):
        return -1.0

    #neumann_functions = None
    neumann_functions = [g_bottom, None, None, None]
else:
    raise ValueError('"domain" must be either "square", "pentagon", or "star".')
mesh_resolution = 50
mesh = MeshGenerator_2d(points, facets, needs_refinement, mesh_resolution)
bcs = BCs_2d(mesh, vertex_values=vertex_values, interpolation_methods=interpolations, neumann_functions=neumann_functions)
solver = ConvectionDiffusion_2d(mesh, f, epsilon=0.8, beta=(0.5, 2.0), alpha=0.0, bcs=bcs)

# Assemble the system and solve
solver.assemble_system()
solver.solve()

# Plot user requests
request = None
while request != 'q':
    try:
        request = input('What would you like to plot and how? Input one word from each of the groups.'
                        'Group 1:\n'
                        '\t- "solution".\n'
                        '\t- "residual".\n'
                        '\t- "source".\n'
                        'Group 2:\n'
                        '\t- "scatter".\n'
                        '\t- "3d".\n'
                        '\t- "contour".\n'
                        'You can also plot the mesh using "mesh".'
                        'Use "q" to quit.\n'
                        'Write your input here: ')

        if request == 'mesh':
            solver.mesh.plot()
            continue

        data_type, plot_style = request.split(' ')
        solver.plot(data_type, plot_style, show=True)
    except ValueError:
        print('Something went wrong there. Try again.')
print('Closing the solver...')
