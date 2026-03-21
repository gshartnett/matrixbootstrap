import numpy as np
from matrixbootstrap.config_utils import generate_configs_bfss, run_all_configs

## energy held fixed
L = 4
checkpoint_path = f"MiniBFSS_L_{L}_symmetric"

config_dir = f"MiniBFSS_L_{L}_symmetric"

#for st_operator_to_minimize in ["x_2", "neg_x_2", "x_4", "neg_x_4"]:
#    for energy in np.linspace(0.4, 3.0, 40):

energy = 1
st_operator_to_minimize = "x_2"

energy = float(np.round(energy, decimals=6))
generate_configs_bfss(
    config_filename=f"energy_{str(energy)}_op_to_min_{st_operator_to_minimize}",
    config_dir=config_dir,
    st_operator_to_minimize=st_operator_to_minimize,
    st_operators_evs_to_set={"energy": energy},
    max_degree_L=L,
    load_from_previously_computed=True,
    checkpoint_path=checkpoint_path,
    impose_symmetries=True,
    optimization_method="newton",
    #optimization_method="pytorch",
    #lr=1e0,
    #init_scale=1e-2,
    )

# execute
run_all_configs(config_dir=config_dir, parallel=True, max_workers=3, check_if_exists_already=False)
