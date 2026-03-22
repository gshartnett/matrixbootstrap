import logging

from matrixbootstrap.config_utils import (
    generate_config_one_matrix,
    run_all_configs,
)

logging.basicConfig(level=logging.INFO)

L = 3
dir = f"OneMatrix_L_{L}_play"

generate_config_one_matrix(
    config_filename="test",
    config_dir=dir,
    checkpoint_dir=dir,
    g2=1,
    g4=1,
    g6=0,
    max_degree_L=3,
    maxiters_cvxpy=10_000_000,
    eps=1e-6,
    load_from_previously_computed=True,
)

# execute
if __name__ == "__main__":
    run_all_configs(config_dir=dir, parallel=False)
