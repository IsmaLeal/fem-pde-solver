import numpy as np
import matplotlib.pyplot as plt
from src.solvers.convection_diffusion_1d import ConvectionDiffusion_1d
from src.mesh.mesh import generate_uniform_mesh


def f(x):
    #return np.sin(5 * np.pi * x) * np.log((x+0.1)**5) - np.tan(6 * x)
    return np.sin(np.pi * x * 2)

# Initialize the FEM solver
N  = 100
domain = (0, 1)
bctype = (0, 1)
bcs = (0, 0)

x_nodes = generate_uniform_mesh(N, domain=domain)
fem_solver = ConvectionDiffusion_1d(x_nodes, epsilon=0.1, beta=0.2, bctype=bctype, bcs=bcs)

new_nodes, u_fem, nodal_history = fem_solver.refine_mesh(f, tol=0.0005, max_iter=5)

# Create grid of evaluation points
N_eval = 400
x_eval = np.linspace(domain[0], domain[1], N_eval)

# Evaluate solution at evaluation points
u_eval = u_fem(x_eval)
u_bvp = fem_solver.solve_scipy(f, x_eval)

# Plot the solution
plt.plot(x_eval, u_eval, label='FEM Solution', color='blue')
plt.plot(x_eval, u_bvp, label='solve_bvp solution', color='red', linestyle='--')
plt.scatter(fem_solver.x_nodes, np.zeros(len(fem_solver.x_nodes)), color='red', s=3)
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Convection-Diffusion FEM and solve_bvp solutions")
plt.legend()
plt.grid(True)
plt.show()
