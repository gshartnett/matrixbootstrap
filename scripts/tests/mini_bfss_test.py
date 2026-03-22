from matrixbootstrap.config_utils import generate_config_bfss, run_all_configs

# generate the config files
L = 3

energy = 1.5
st_operator_to_minimize = "x_2"

generate_config_bfss(
    config_filename="test",
    config_dir=f"MiniBFSS_L_{L}_test",
    checkpoint_path=f"MiniBFSS_L_{L}_symmetric",
    max_degree_L=L,
    load_from_previously_computed=True,
    odd_degree_vanish=True,
    simplify_quadratic=True,
    st_operator_to_minimize=st_operator_to_minimize,
    st_operators_evs_to_set={"energy": energy},
    #optimization_method="pytorch",
    optimization_method="newton",
    cvxpy_solver='MOSEK',
    reg=1e-4,
    )

# execute
run_all_configs(
    config_dir=f"MiniBFSS_L_{L}_test",
    parallel=False,
    check_if_exists_already=False
    )