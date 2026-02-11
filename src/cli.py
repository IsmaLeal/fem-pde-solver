import argparse
import numpy as np
from .fem_solver import solve_fem


def parse_function(func_str):
    """
    Parses a mathematical function string into a Python function.

    Example:
        Input: "sin(pi*x)*cos(pi*y)"
        Output: lambda x, y: np.sin(np.pi*x) * np.cos(np.pi*y)
    """

    try:
        if not func_str:
            return eval(f"lambda x, y: np.zeros_like(x, dtype=float)")
        else:
            return eval(f"lambda x, y: {func_str}", {"np": np,
                                                     "sin": np.sin,
                                                     "cos": np.cos,
                                                     "pi": np.pi,
                                                     "e":np.e})
    except Exception as e:
        print(f"Invalid function format: {e}")
        exit(1)


def parse_list(input_str, dtype=float):
    """Parses a string like '0,0;1,0;1,1' into a list of tuples [(0, 0), (1, 0), (1, 1)]"""
    if not input_str:
        return None
    try:
        if dtype == tuple:
            return [tuple(map(float, item.split(','))) for item in input_str.split(';')]
        elif dtype == float:
            return [float(item) for item in input_str.split(';')]
        elif dtype == str:
            return input_str.split(';')
    except ValueError:
        raise ValueError(f"Invalid format. Expected a list of {dtype.__name__}. Use ';' as separator.")


def main():
    parser = argparse.ArgumentParser(description="2D FEM Solver CLI")

    # Domain selection
    parser.add_argument('--domain', type=str, choices=['square', 'star', 'pentagon'], default=None,
                        help="Predefined domain (optional). Leave empty for custom domain.")

    # Custom geometry
    parser.add_argument('--points', type=str, help='Custom domain points as "x1,y1;x2,y2;...".')
    parser.add_argument('--facets', type=str, help="Custom facets as 'i1,i2;i2,i3;...'.")
    parser.add_argument('--vertex_values', type=str, help="Vertex values as 'v1;v2;v3;...' (one per point).")
    parser.add_argument('--interpolations', type=str, help="Interpolation methods as 'linear;quadratic;...' (one per facet).")
    parser.add_argument('--interpolation_func', type=str, default=None, help='Custom interpolation function in terms of t, g1, g2 (e.g., "sin(pi*t) + t")')


    # PDE selection
    parser.add_argument('--pde', type=str, choices=['poisson', 'convection_diffusion'], required=True,
                        help="Type of PDE to solve.")

    # FEM parameters
    parser.add_argument('--resolution', type=int, default=50, help="Mesh resolution (default: 50).")
    parser.add_argument('--epsilon', type=float, default=0.8, help="Diffusion coefficient for convection-diffusion.")
    parser.add_argument('--beta', type=float, nargs=2, default=[0.0, 0.0], help="Convection coefficients as two floats.")
    parser.add_argument('--alpha', type=float, default=0.0, help="Reaction coefficient for convection-diffusion.")
    parser.add_argument('--max_triangle_area', type=float, default=0.01, help="Maximum area of the mesh triangles.")
    parser.add_argument('--f', type=str, help='Forcing function in terms of x and y.')

    args = parser.parse_args()

    # Parse inputs
    f = parse_function(args.f)
    points = parse_list(args.points, dtype=tuple) if args.points else None
    facets = parse_list(args.facets, dtype=tuple) if args.facets else None
    if facets:
        facets = np.array(facets).astype(int)
    vertex_values = parse_list(args.vertex_values, dtype=float) if args.vertex_values else None
    interpolations = parse_list(args.interpolations, dtype=str) if args.interpolations else None

    solver = solve_fem(args.domain, args.pde, args.resolution, args.epsilon, args.beta,
                       args.alpha, args.max_triangle_area, f, points, facets,
                       vertex_values, interpolations)

    # CLI plotting
    request = input("Enter a command ('q' to quit, 'mesh' to plot mesh, 'solution contour', etc.): ").strip().lower()
    while request != 'q':
        try:
            request = input('(i) In order to plot something, input one word from each of the groups.\n'
                            '\tGroup 1:\n'
                            '\t\t- "solution".\n'
                            '\t\t- "residual".\n'
                            '\t\t- "source".\n'
                            '\tGroup 2:\n'
                            '\t\t- "scatter", only accepted after "solution".\n'
                            '\t\t- "3d".\n'
                            '\t\t- "contour".\n'
                            '\t----------------------\n'
                            '\tExample inputs:\n'
                            '\t\t- "solution scatter"\n'
                            '\t\t- "residual contour\n"'
                            '\t\t- "source 3d"\n'
                            '\t\t- "solution 3d"\n'
                            '\t----------------------\n\n'
                            '\tYou can also plot the mesh by typing "mesh".\n\n'
                            '(ii) You can compute and print one of the following quantities:\n'
                            '\t- "residual norm".\n'
                            '\t- "source norm" '
                            r'for the L2-norm of the source term $||f(x, y)||_{L_2}$.'
                            '\nUse "q" to quit.\n'
                            'Write your input here: ')

            if request == 'mesh':
                solver.mesh.plot()
                continue
            elif request == 'residual norm':
                print(solver.get_residual_norm())
                continue
            elif request == 'source norm':
                print(solver.get_rhs_norm())
                continue
            else:
                data_type, plot_style = request.split(' ')
                solver.plot(data_type, plot_style, show=True)
        except ValueError:
            print('\n\nOops... Something went wrong there! :(\n\n At least you can try again :)!\n\n')
    print('Closing the solver...')

if __name__ == "__main__":
    main()
