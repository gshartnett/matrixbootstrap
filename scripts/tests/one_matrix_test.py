from matrixbootstrap.brezin import compute_Brezin_energy
from matrixbootstrap.config_utils import generate_config_one_matrix, run_all_configs

L = 3
g2 = 1
g4 = 1
g6 = 0

generate_config_one_matrix(
    config_filename="test",
    config_dir=f"OneMatrix_L_{L}_test",
    g2=g2,
    g4=g4,
    g6=g6,
    max_degree_L=L,
    impose_symmetries=False,
    load_from_previously_computed=False,
    odd_degree_vanish=True,
    simplify_quadratic=True,
    optimization_method="newton",
    )

# execute
run_all_configs(config_dir=f"OneMatrix_L_{L}_test", parallel=False, check_if_exists_already=False)
print(f"Brezin energy = {compute_Brezin_energy(g=g4/4)}")