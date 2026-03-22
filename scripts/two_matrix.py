import logging

import numpy as np

from matrixbootstrap.config_utils import (
    generate_config_two_matrix,
    run_all_configs,
)

logging.basicConfig(level=logging.INFO)

## energy held fixed
L = 3
g2 = 1

config_dir = f"TwoMatrix_L_{L}_symmetric_energy_fixed_g2_{g2}"
# config_dir = f"TwoMatrix_L_{L}_symmetric_energy_fixed_g2_{g2}_pytorch"
# config_dir = f"TwoMatrix_L_{L}_symmetric_energy_fixed_g2_{g2}_newton_Axb"

for st_operator_to_minimize in ["x_2", "neg_x_2"]:
    for energy in np.linspace(0.9, 1.4, 81):
        energy = float(np.round(energy, decimals=6))
        generate_config_two_matrix(
            config_dir=config_dir,
            g2=g2,
            g4=1,
            st_operator_to_minimize=st_operator_to_minimize,
            st_operators_evs_to_set={"energy": energy},
            max_degree_L=L,
            load_from_previously_computed=True,
            impose_symmetries=True,
            optimization_method="newton",
            cvxpy_solver="SCS",
            maxiters=30,
            init_scale=1e-2,
            reg=1e-4,
            penalty_reg=0,
            tol=1e-7,
        )

# execute
if __name__ == "__main__":
    run_all_configs(
        config_dir=config_dir,
        parallel=True,
        max_workers=6,
        check_if_exists_already=True,
    )
