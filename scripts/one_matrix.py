import numpy as np
from matrixbootstrap.config_utils import generate_configs_one_matrix, run_all_configs

n_grid = 20
g4_max = 16
g6_max = 16
g2_values = [-1, 0, 1]
g4_values = np.concatenate((np.linspace(-g4_max, 0, n_grid), np.linspace(0, g4_max, n_grid)[1:]))
g6_values = np.linspace(0, g6_max, n_grid)

for L in [3, 4]:

    # generate the config files
    for g2 in g2_values:
        for g4 in g4_values:
            for g6 in g6_values:

                g4 = float(np.round(g4, decimals=6))
                g6 = float(np.round(g6, decimals=6))

                # only run models with bounded-by-below potentials
                if (g6 > 0) or (g6==0 and g4 > 0):

                    generate_configs_one_matrix(
                        config_filename=f"g2_{str(g2)}_g4_{str(g4)}_g6_{str(g6)}",
                        config_dir=f"OneMatrix_L_{L}",
                        g2=g2,
                        g4=g4,
                        g6=g6,
                        max_degree_L=L,
                        maxiters_cvxpy=5_000,
                        maxiters=100,
                        radius=1e6,
                        #reg=1e6,
                        )

    # execute
    run_all_configs(config_dir=f"OneMatrix_L_{L}", parallel=True)