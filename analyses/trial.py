import numpy as np
import matplotlib.pyplot as plt
from src.solvers.convection_diffusion_1d import ConvectionDiffusionFEM
from src.mesh.mesh import generate_uniform_mesh

vals = [10, 1, 1 / np.pi, 1 / (10 * np.pi)]
vals_names = [r'$f(x) = \sin(5\pi x)\log(x^5) - \tan(x / 10)$',
              r'$f(x) = \sin(5\pi x)\log(x^5) - \tan(x)$',
              r'$f(x) = \sin(5\pi x)\log(x^5) - \tan(\pi x)$',
              r'$f(x) = \sin(5\pi x)\log(x^5) - \tan(10\pi x)$',]

fig, axs = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(r'FEM and solve_bvp solutions for different forcing terms $f(x)$')

for idx, val in enumerate(vals):
    def f(x):
        return np.sin(5 * np.pi * x) * np.log((x+0.01)**5) - np.tan(x / val)

    # Initialize the FEM solver
    N  = 100
    domain = (0, 4)
    x_nodes = generate_uniform_mesh(N, domain=domain)

    bctype = (0, 1)
    bcs = (4, -10)
    fem_solver = ConvectionDiffusionFEM(x_nodes, epsilon=0.1, beta=20, bctype=bctype, bcs=bcs)

    new_nodes, u_fem, nodal_history = fem_solver.refine_mesh_1d(f, tol=0.0005, max_iter=7)

    # Create grid of evaluation points
    N_eval = 400
    x_eval = np.linspace(domain[0], domain[1], N_eval)

    # Evaluate solution at evaluation points
    u_eval = u_fem(x_eval)
    u_bvp = fem_solver.solve_scipy(f, x_eval)

    min_value = np.min([np.min(u_eval), np.min(u_bvp)])
    if idx == 2:
        min_value = 3.86
    min_vals = np.zeros(len(fem_solver.x_nodes)) + min_value

    # Plot the solution
    axs[int(round(idx / (len(vals) - 1))), int(idx % 2)].plot(x_eval, u_eval, label='FEM Solution', color='blue')
    axs[int(round(idx / (len(vals) - 1))), int(idx % 2)].plot(x_eval, u_bvp, label='solve_bvp solution', color='red', linestyle='--')
    axs[int(round(idx / (len(vals) - 1))), int(idx % 2)].scatter(fem_solver.x_nodes, min_vals, color='red', s=1.5)
    axs[int(round(idx / (len(vals) - 1))), int(idx % 2)].set_xlabel('x')
    axs[int(round(idx / (len(vals) - 1))), int(idx % 2)].set_ylabel('u(x)')
    axs[int(round(idx / (len(vals) - 1))), int(idx % 2)].set_title(vals_names[idx])
    axs[int(round(idx / (len(vals) - 1))), int(idx % 2)].legend()
    axs[int(round(idx / (len(vals) - 1))), int(idx % 2)].grid(True)

    if idx > 1:
        axs[int(round(idx / (len(vals) - 1))), int(idx % 2)].set_ylim(min(u_eval) - 0.5, max(u_eval) + 0.5)

axs[1, 1].set_ylim(3.85, 4.1)
plt.show()
