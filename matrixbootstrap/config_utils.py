import os
import json
import yaml
import fire
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from matrixbootstrap.algebra import SingleTraceOperator
from matrixbootstrap.bootstrap import BootstrapSystem
from matrixbootstrap.bootstrap_complex import BootstrapSystemComplex
from matrixbootstrap.models import OneMatrix, TwoMatrix, MiniBFSS, ThreeMatrix, MiniBMN
from matrixbootstrap.solver_newton import solve_bootstrap as solve_bootstrap_newton
from matrixbootstrap.solver_pytorch import solve_bootstrap as solve_bootstrap_pytorch


bootstrap_keys = [
    "max_degree_L",
    "st_operator_to_minimize",
    "st_operators_evs_to_set",
    "odd_degree_vanish",
    "simplify_quadratic",
    "impose_symmetries",
    "symmetry_method",
    "impose_gauge_symmetry",
    "load_from_previously_computed",
    "checkpoint_path",
    ]

optimization_keys_newton=[
    "init_scale",
    "init",
    "maxiters",
    "maxiters_cvxpy",
    "cvxpy_solver",
    "tol",
    'reg',
    "eps_abs",
    "eps_rel",
    "eps_infeas",
    "radius",
    "PRNG_seed",
    ]

optimization_keys_pytorch=[
    "init_scale",
    "init",
    "PRNG_seed",
    "lr",
    "gamma",
    "num_epochs",
    "penalty_reg",
    "patience",
    ]

def generate_optimization_config_newton(
    PRNG_seed=None,
    init=None,
    init_scale=1e-2,
    maxiters=100,
    maxiters_cvxpy=10_000,
    tol=1e-7,
    reg=1e-4,
    eps_abs=1e-5,
    eps_rel=1e-5,
    eps_infeas=1e-7,
    radius=1e5,
    cvxpy_solver='SCS',
    ):

    optimization_config_dict={
        "init": init,
        "PRNG_seed": PRNG_seed,
        "init_scale": init_scale,
        "maxiters": maxiters,
        "maxiters_cvxpy": maxiters_cvxpy,
        "tol": tol,
        "reg": reg,
        "eps_abs": eps_abs,
        "eps_rel": eps_rel,
        "eps_infeas": eps_infeas,
        "radius": radius,
        "cvxpy_solver": cvxpy_solver,
        "optimization_method": "newton",
        }

    return optimization_config_dict


def generate_optimization_config_pytorch(
    PRNG_seed=None,
    init=None,
    init_scale=1e-2,
    lr=1e-2,
    gamma=0.9995,
    num_epochs=30_000,
    penalty_reg=1e2,
    tol=1e-8,
    patience=100,
    early_stopping_tol=1e-3,
    ):

    optimization_config_dict={
        "init": init,
        "PRNG_seed": PRNG_seed,
        "init_scale": init_scale,
        "lr": lr,
        "gamma": gamma,
        "num_epochs": num_epochs,
        "penalty_reg": penalty_reg,
        "optimization_method": "pytorch",
        "tol": tol,
        "patience": patience,
        "early_stopping_tol": early_stopping_tol
        }

    return optimization_config_dict


def generate_bootstrap_config(
    max_degree_L=3,
    st_operator_to_minimize="energy",
    st_operators_evs_to_set=None,
    odd_degree_vanish=True,
    simplify_quadratic=True,
    impose_symmetries=True,
    load_from_previously_computed=False,
    impose_gauge_symmetry=True,
    checkpoint_path=None,
    symmetry_method="complete",
    ):

    bootstrap_config_dict = {
        "max_degree_L": max_degree_L,
        "st_operator_to_minimize": st_operator_to_minimize,
        "st_operators_evs_to_set": st_operators_evs_to_set,
        "odd_degree_vanish": odd_degree_vanish,
        "simplify_quadratic": simplify_quadratic,
        "impose_symmetries": impose_symmetries,
        "symmetry_method": symmetry_method,
        "impose_gauge_symmetry": impose_gauge_symmetry,
        "load_from_previously_computed": load_from_previously_computed,
        "checkpoint_path": checkpoint_path,
        }

    return bootstrap_config_dict


def generate_config_one_matrix(
    config_filename,
    config_dir,
    g2,
    g4,
    g6,
    optimization_method,
    **kwargs):

    if not optimization_method in ["newton", "pytorch"]:
        raise ValueError(f"optimization method {optimization_method} not recognized.")

    # split the kwargs into separate bootstrap and optimization kwargs
    kwargs_bootstrap = {key: kwargs[key] for key in bootstrap_keys if key in kwargs}
    if optimization_method == "newton":
        kwargs_optimization = {key: kwargs[key] for key in optimization_keys_newton if key in kwargs}
    elif optimization_method == "pytorch":
        kwargs_optimization = {key: kwargs[key] for key in optimization_keys_pytorch if key in kwargs}

    # build the bootstrap and optimization configs
    bootstrap_config_dict = generate_bootstrap_config(**kwargs_bootstrap)
    if optimization_method == "newton":
        optimization_config_dict = generate_optimization_config_newton(**kwargs_optimization)
    elif optimization_method == "pytorch":
        optimization_config_dict = generate_optimization_config_pytorch(**kwargs_optimization)

    # build the config dictionary
    config_data = {
        "model": {
            "model name": "OneMatrix",
            "bootstrap class": "BootstrapSystem",
            "couplings": {"g2": float(g2), "g4": float(g4), "g6": float(g6)},
        },
        "bootstrap": bootstrap_config_dict,
        "optimizer": optimization_config_dict,
    }

    # write to yaml
    #if not os.path.exists(f"configs/{config_dir}"):
    os.makedirs(f"configs/{config_dir}", exist_ok=True)
    with open(f"configs/{config_dir}/{config_filename}.yaml", "w") as outfile:
        yaml.dump(config_data, outfile, default_flow_style=False)

    return


def generate_config_two_matrix(
    config_filename,
    config_dir,
    g2,
    g4,
    optimization_method,
    **kwargs):

    if not optimization_method in ["newton", "pytorch"]:
        raise ValueError(f"optimization method {optimization_method} not recognized.")

    # split the kwargs into separate bootstrap and optimization kwargs
    kwargs_bootstrap = {key: kwargs[key] for key in bootstrap_keys if key in kwargs}
    if optimization_method == "newton":
        kwargs_optimization = {key: kwargs[key] for key in optimization_keys_newton if key in kwargs}
    elif optimization_method == "pytorch":
        kwargs_optimization = {key: kwargs[key] for key in optimization_keys_pytorch if key in kwargs}

    # build the bootstrap and optimization configs
    bootstrap_config_dict = generate_bootstrap_config(**kwargs_bootstrap)
    if optimization_method == "newton":
        optimization_config_dict = generate_optimization_config_newton(**kwargs_optimization)
    elif optimization_method == "pytorch":
        optimization_config_dict = generate_optimization_config_pytorch(**kwargs_optimization)

    # build the config dictionary
    config_data = {
        "model": {
            "model name": "TwoMatrix",
            "bootstrap class": "BootstrapSystem",
            "couplings": {"g2": float(g2), "g4": float(g4)},
        },
        "bootstrap": bootstrap_config_dict,
        "optimizer": optimization_config_dict,
    }

    # write to yaml
    #if not os.path.exists(f"configs/{config_dir}"):
    os.makedirs(f"configs/{config_dir}", exist_ok=True)
    with open(f"configs/{config_dir}/{config_filename}.yaml", "w") as outfile:
        yaml.dump(config_data, outfile, default_flow_style=False)

    return


def generate_config_three_matrix(
    config_filename,
    config_dir,
    g2,
    g3,
    g4,
    optimization_method,
    **kwargs):

    if not optimization_method in ["newton", "pytorch"]:
        raise ValueError(f"optimization method {optimization_method} not recognized.")

    # split the kwargs into separate bootstrap and optimization kwargs
    kwargs_bootstrap = {key: kwargs[key] for key in bootstrap_keys if key in kwargs}
    if optimization_method == "newton":
        kwargs_optimization = {key: kwargs[key] for key in optimization_keys_newton if key in kwargs}
    elif optimization_method == "pytorch":
        kwargs_optimization = {key: kwargs[key] for key in optimization_keys_pytorch if key in kwargs}

    # build the bootstrap and optimization configs
    bootstrap_config_dict = generate_bootstrap_config(**kwargs_bootstrap)
    if optimization_method == "newton":
        optimization_config_dict = generate_optimization_config_newton(**kwargs_optimization)
    elif optimization_method == "pytorch":
        optimization_config_dict = generate_optimization_config_pytorch(**kwargs_optimization)

    # build the config dictionary
    config_data = {
        "model": {
            "model name": "ThreeMatrix",
            "bootstrap class": "BootstrapSystem",
            "couplings": {"g2": float(g2), "g3": float(g3), "g4": float(g4)},
        },
        "bootstrap": bootstrap_config_dict,
        "optimizer": optimization_config_dict,
    }

    # write to yaml
    #if not os.path.exists(f"configs/{config_dir}"):
    os.makedirs(f"configs/{config_dir}", exist_ok=True)
    with open(f"configs/{config_dir}/{config_filename}.yaml", "w") as outfile:
        yaml.dump(config_data, outfile, default_flow_style=False)

    return


def generate_config_bfss(
    config_filename,
    config_dir,
    optimization_method,
    **kwargs):

    if not optimization_method in ["newton", "pytorch"]:
        raise ValueError(f"optimization method {optimization_method} not recognized.")

    # split the kwargs into separate bootstrap and optimization kwargs
    kwargs_bootstrap = {key: kwargs[key] for key in bootstrap_keys if key in kwargs}
    if optimization_method == "newton":
        kwargs_optimization = {key: kwargs[key] for key in optimization_keys_newton if key in kwargs}
    elif optimization_method == "pytorch":
        kwargs_optimization = {key: kwargs[key] for key in optimization_keys_pytorch if key in kwargs}

    # build the bootstrap and optimization configs
    bootstrap_config_dict = generate_bootstrap_config(**kwargs_bootstrap)
    if optimization_method == "newton":
        optimization_config_dict = generate_optimization_config_newton(**kwargs_optimization)
    elif optimization_method == "pytorch":
        optimization_config_dict = generate_optimization_config_pytorch(**kwargs_optimization)

    # build the config dictionary
    config_data = {
        "model": {
            "model name": "MiniBFSS",
            "bootstrap class": "BootstrapSystem",
            "couplings": {"lambda": 1},
        },
        "bootstrap": bootstrap_config_dict,
        "optimizer": optimization_config_dict,
    }

    # write to yaml
    #if not os.path.exists(f"configs/{config_dir}"):
    os.makedirs(f"configs/{config_dir}", exist_ok=True)
    with open(f"configs/{config_dir}/{config_filename}.yaml", "w") as outfile:
        yaml.dump(config_data, outfile, default_flow_style=False)

    return

def generate_config_bmn(
    config_filename,
    config_dir,
    nu,
    lambd,
    optimization_method,
    **kwargs):

    if not optimization_method in ["newton", "pytorch"]:
        raise ValueError(f"optimization method {optimization_method} not recognized.")

    # split the kwargs into separate bootstrap and optimization kwargs
    kwargs_bootstrap = {key: kwargs[key] for key in bootstrap_keys if key in kwargs}
    if optimization_method == "newton":
        kwargs_optimization = {key: kwargs[key] for key in optimization_keys_newton if key in kwargs}
    elif optimization_method == "pytorch":
        kwargs_optimization = {key: kwargs[key] for key in optimization_keys_pytorch if key in kwargs}

    # build the bootstrap and optimization configs
    bootstrap_config_dict = generate_bootstrap_config(**kwargs_bootstrap)
    if optimization_method == "newton":
        optimization_config_dict = generate_optimization_config_newton(**kwargs_optimization)
    elif optimization_method == "pytorch":
        optimization_config_dict = generate_optimization_config_pytorch(**kwargs_optimization)

    # build the config dictionary
    config_data = {
        "model": {
            "model name": "MiniBMN",
            #"bootstrap class": "BootstrapSystemComplex",
            "bootstrap class": "BootstrapSystem",
            "couplings": {"nu": float(nu), "lambda": float(lambd)},
        },
        "bootstrap": bootstrap_config_dict,
        "optimizer": optimization_config_dict,
    }

    # write to yaml
    #if not os.path.exists(f"configs/{config_dir}"):
    os.makedirs(f"configs/{config_dir}", exist_ok=True)
    with open(f"configs/{config_dir}/{config_filename}.yaml", "w") as outfile:
        yaml.dump(config_data, outfile, default_flow_style=False)

    return


def run_bootstrap_from_config(config_filename, config_dir, verbose=True, check_if_exists_already=True):

    # optionally skip if data file already exists
    if check_if_exists_already:
        print(f"data/{config_dir}/{config_filename}.json")
        if os.path.exists(f"data/{config_dir}/{config_filename}.json"):
            print(f"Run result already exists, skipping.")
            return

    # load the config file
    with open(f"configs/{config_dir}/{config_filename}.yaml") as stream:
        config = yaml.safe_load(stream)
    config_model = config["model"]
    config_bootstrap = config["bootstrap"]
    config_optimizer = config["optimizer"]

    # build the model
    model = globals()[config_model["model name"]](couplings=config_model["couplings"])

    # checkpoint path
    if config_bootstrap["checkpoint_path"] is None:
        checkpoint_path = f"checkpoints/" + config_model["model name"] + "_L_" + str(config_bootstrap["max_degree_L"])
    else:
        checkpoint_path = "checkpoints/" + config_bootstrap["checkpoint_path"]

    # handle the imposition of global symmetries
    if not config_bootstrap["impose_symmetries"]:
        model.symmetry_generators = None

    # handle the imposition of gauge symmetries
    if not config_bootstrap["impose_gauge_symmetry"]:
        model.gauge_generator = None

    # operator to minimize
    if config_bootstrap["st_operator_to_minimize"] is None:
        st_operator_to_minimize = None
    else:
        st_operator_to_minimize = model.operators_to_track[config_bootstrap["st_operator_to_minimize"]]

    # operators whose expectation values are to be fixed
    st_operator_inhomo_constraints=[(SingleTraceOperator(data={(): 1}), 1)]
    if config_bootstrap["st_operators_evs_to_set"] is not None:
        for key, value in config_bootstrap["st_operators_evs_to_set"].items():
            st_operator_inhomo_constraints.append(
                (model.operators_to_track[key], value)
            )

    # build the bootstrap
    bootstrap = globals()[config_model["bootstrap class"]](
        matrix_system=model.matrix_system,
        hamiltonian=model.hamiltonian,
        gauge_generator=model.gauge_generator,
        max_degree_L=config_bootstrap["max_degree_L"],
        odd_degree_vanish=config_bootstrap["odd_degree_vanish"],
        simplify_quadratic=config_bootstrap["simplify_quadratic"],
        symmetry_generators=model.symmetry_generators,
        #impose_gauge_symmetry=config_bootstrap["impose_gauge_symmetry"],
        checkpoint_path=checkpoint_path,
    )

    # load previously-computed constraints
    if config_bootstrap["load_from_previously_computed"] and os.path.exists(checkpoint_path):
        bootstrap.load_constraints(checkpoint_path)

    # solve the bootstrap
    optimization_method = config_optimizer.pop("optimization_method")
    if optimization_method == "newton":
        param, optimization_result = solve_bootstrap_newton(
            bootstrap=bootstrap,
            st_operator_to_minimize=st_operator_to_minimize,
            st_operator_inhomo_constraints=st_operator_inhomo_constraints,
            **config_optimizer
            )
    elif optimization_method == "pytorch":
        param, optimization_result = solve_bootstrap_pytorch(
            bootstrap=bootstrap,
            st_operator_to_minimize=st_operator_to_minimize,
            st_operator_inhomo_constraints=st_operator_inhomo_constraints,
            **config_optimizer
            )

    if param is None:
        return

    # record select expectation values
    expectation_values = {
        name: float(bootstrap.get_operator_expectation_value(st_operator=st_operator, param=param).real)
        for name, st_operator in model.operators_to_track.items()
        }

    # save the results
    result = optimization_result | expectation_values
    result["param"] = list(param)

    #if not os.path.exists(f"data/{config_dir}"):
    os.makedirs(f"data/{config_dir}", exist_ok=True)
    with open(f"data/{config_dir}/{config_filename}.json", "w") as f:
        json.dump(result, f)

    for key, value in expectation_values.items():
        print(f"EV {key}: {value:.4e}")

    return result


def run_all_configs(config_dir, parallel=False, max_workers=6, verbose=True, check_if_exists_already=True):

    config_filenames = os.listdir(f"configs/{config_dir}")
    config_filenames = [f[:-5] for f in config_filenames if '.yaml' in f]
    #np.random.shuffle(config_filenames) # shuffle

    if not parallel:
        for config_filename in config_filenames:
            run_bootstrap_from_config(config_filename, config_dir, check_if_exists_already=check_if_exists_already)
    else:
        with ProcessPoolExecutor(max_workers) as executor:
            futures = [executor.submit(run_bootstrap_from_config, config_filename, config_dir, verbose, check_if_exists_already) for config_filename in config_filenames]
        for future in futures:
            future.result()
        print('finished!')


if __name__ == "__main__":
    fire.Fire()