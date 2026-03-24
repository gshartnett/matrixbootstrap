import hashlib
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor

import fire
import yaml

from matrixbootstrap.algebra import SingleTraceOperator
from matrixbootstrap.bootstrap import BootstrapSystem
from matrixbootstrap.bootstrap_complex import BootstrapSystemComplex
from matrixbootstrap.models import (
    MiniBFSS,
    MiniBMN,
    OneMatrix,
    ThreeMatrix,
    TwoMatrix,
)
from matrixbootstrap.solver_newton import solve_bootstrap as solve_bootstrap_newton
from matrixbootstrap.solver_pytorch import solve_bootstrap as solve_bootstrap_pytorch

logger = logging.getLogger(__name__)

_MODEL_CLASSES = {
    "OneMatrix": OneMatrix,
    "TwoMatrix": TwoMatrix,
    "ThreeMatrix": ThreeMatrix,
    "MiniBFSS": MiniBFSS,
    "MiniBMN": MiniBMN,
}

_BOOTSTRAP_CLASSES = {
    "BootstrapSystem": BootstrapSystem,
    "BootstrapSystemComplex": BootstrapSystemComplex,
}

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
]

optimization_keys_newton = [
    "init_scale",
    "init",
    "maxiters",
    "maxiters_cvxpy",
    "cvxpy_solver",
    "tol",
    "reg",
    "eps_abs",
    "eps_rel",
    "eps_infeas",
    "radius",
    "PRNG_seed",
    "use_factorization_block",
]

optimization_keys_pytorch = [
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
    cvxpy_solver="SCS",
    use_factorization_block=False,
):

    optimization_config_dict = {
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
        "use_factorization_block": use_factorization_block,
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

    optimization_config_dict = {
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
        "early_stopping_tol": early_stopping_tol,
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
    }

    return bootstrap_config_dict


def _config_id(config_data: dict) -> str:
    """Generate a short deterministic hex ID from config content."""
    content = json.dumps(config_data, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def _struct_hash(config_data: dict) -> str:
    """
    Generate a hex ID from the coupling-independent structural parameters.

    The structural cache (cyclic quadratic constraints) depends only on the
    model type and max_degree_L, not on coupling values or optimizer settings.
    """
    structural = {
        "model_name": config_data["model"]["model name"],
        "bootstrap_class": config_data["model"]["bootstrap class"],
        "max_degree_L": config_data["bootstrap"]["max_degree_L"],
    }
    content = json.dumps(structural, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def generate_config_one_matrix(config_dir, g2, g4, g6, optimization_method, **kwargs):

    if optimization_method not in ["newton", "pytorch"]:
        raise ValueError(f"optimization method {optimization_method} not recognized.")

    # split the kwargs into separate bootstrap and optimization kwargs
    kwargs_bootstrap = {key: kwargs[key] for key in bootstrap_keys if key in kwargs}
    if optimization_method == "newton":
        kwargs_optimization = {
            key: kwargs[key] for key in optimization_keys_newton if key in kwargs
        }
    elif optimization_method == "pytorch":
        kwargs_optimization = {
            key: kwargs[key] for key in optimization_keys_pytorch if key in kwargs
        }

    # build the bootstrap and optimization configs
    bootstrap_config_dict = generate_bootstrap_config(**kwargs_bootstrap)
    if optimization_method == "newton":
        optimization_config_dict = generate_optimization_config_newton(
            **kwargs_optimization
        )
    elif optimization_method == "pytorch":
        optimization_config_dict = generate_optimization_config_pytorch(
            **kwargs_optimization
        )

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

    config_id = _config_id(config_data)
    os.makedirs(f"runs/{config_dir}/configs", exist_ok=True)
    with open(f"runs/{config_dir}/configs/{config_id}.yaml", "w") as outfile:
        yaml.dump(config_data, outfile, default_flow_style=False)

    return config_id


def generate_config_two_matrix(config_dir, g2, g4, optimization_method, **kwargs):

    if optimization_method not in ["newton", "pytorch"]:
        raise ValueError(f"optimization method {optimization_method} not recognized.")

    # split the kwargs into separate bootstrap and optimization kwargs
    kwargs_bootstrap = {key: kwargs[key] for key in bootstrap_keys if key in kwargs}
    if optimization_method == "newton":
        kwargs_optimization = {
            key: kwargs[key] for key in optimization_keys_newton if key in kwargs
        }
    elif optimization_method == "pytorch":
        kwargs_optimization = {
            key: kwargs[key] for key in optimization_keys_pytorch if key in kwargs
        }

    # build the bootstrap and optimization configs
    bootstrap_config_dict = generate_bootstrap_config(**kwargs_bootstrap)
    if optimization_method == "newton":
        optimization_config_dict = generate_optimization_config_newton(
            **kwargs_optimization
        )
    elif optimization_method == "pytorch":
        optimization_config_dict = generate_optimization_config_pytorch(
            **kwargs_optimization
        )

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

    config_id = _config_id(config_data)
    os.makedirs(f"runs/{config_dir}/configs", exist_ok=True)
    with open(f"runs/{config_dir}/configs/{config_id}.yaml", "w") as outfile:
        yaml.dump(config_data, outfile, default_flow_style=False)

    return config_id


def generate_config_three_matrix(config_dir, g2, g3, g4, optimization_method, **kwargs):

    if optimization_method not in ["newton", "pytorch"]:
        raise ValueError(f"optimization method {optimization_method} not recognized.")

    # split the kwargs into separate bootstrap and optimization kwargs
    kwargs_bootstrap = {key: kwargs[key] for key in bootstrap_keys if key in kwargs}
    if optimization_method == "newton":
        kwargs_optimization = {
            key: kwargs[key] for key in optimization_keys_newton if key in kwargs
        }
    elif optimization_method == "pytorch":
        kwargs_optimization = {
            key: kwargs[key] for key in optimization_keys_pytorch if key in kwargs
        }

    # build the bootstrap and optimization configs
    bootstrap_config_dict = generate_bootstrap_config(**kwargs_bootstrap)
    if optimization_method == "newton":
        optimization_config_dict = generate_optimization_config_newton(
            **kwargs_optimization
        )
    elif optimization_method == "pytorch":
        optimization_config_dict = generate_optimization_config_pytorch(
            **kwargs_optimization
        )

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

    config_id = _config_id(config_data)
    os.makedirs(f"runs/{config_dir}/configs", exist_ok=True)
    with open(f"runs/{config_dir}/configs/{config_id}.yaml", "w") as outfile:
        yaml.dump(config_data, outfile, default_flow_style=False)

    return config_id


def generate_config_bfss(config_dir, optimization_method, **kwargs):

    if optimization_method not in ["newton", "pytorch"]:
        raise ValueError(f"optimization method {optimization_method} not recognized.")

    # split the kwargs into separate bootstrap and optimization kwargs
    kwargs_bootstrap = {key: kwargs[key] for key in bootstrap_keys if key in kwargs}
    if optimization_method == "newton":
        kwargs_optimization = {
            key: kwargs[key] for key in optimization_keys_newton if key in kwargs
        }
    elif optimization_method == "pytorch":
        kwargs_optimization = {
            key: kwargs[key] for key in optimization_keys_pytorch if key in kwargs
        }

    # build the bootstrap and optimization configs
    bootstrap_config_dict = generate_bootstrap_config(**kwargs_bootstrap)
    if optimization_method == "newton":
        optimization_config_dict = generate_optimization_config_newton(
            **kwargs_optimization
        )
    elif optimization_method == "pytorch":
        optimization_config_dict = generate_optimization_config_pytorch(
            **kwargs_optimization
        )

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

    config_id = _config_id(config_data)
    os.makedirs(f"runs/{config_dir}/configs", exist_ok=True)
    with open(f"runs/{config_dir}/configs/{config_id}.yaml", "w") as outfile:
        yaml.dump(config_data, outfile, default_flow_style=False)

    return config_id


def generate_config_bmn(config_dir, nu, lambd, optimization_method, **kwargs):

    if optimization_method not in ["newton", "pytorch"]:
        raise ValueError(f"optimization method {optimization_method} not recognized.")

    # split the kwargs into separate bootstrap and optimization kwargs
    kwargs_bootstrap = {key: kwargs[key] for key in bootstrap_keys if key in kwargs}
    if optimization_method == "newton":
        kwargs_optimization = {
            key: kwargs[key] for key in optimization_keys_newton if key in kwargs
        }
    elif optimization_method == "pytorch":
        kwargs_optimization = {
            key: kwargs[key] for key in optimization_keys_pytorch if key in kwargs
        }

    # build the bootstrap and optimization configs
    bootstrap_config_dict = generate_bootstrap_config(**kwargs_bootstrap)
    if optimization_method == "newton":
        optimization_config_dict = generate_optimization_config_newton(
            **kwargs_optimization
        )
    elif optimization_method == "pytorch":
        optimization_config_dict = generate_optimization_config_pytorch(
            **kwargs_optimization
        )

    # build the config dictionary
    config_data = {
        "model": {
            "model name": "MiniBMN",
            # "bootstrap class": "BootstrapSystemComplex",
            "bootstrap class": "BootstrapSystem",
            "couplings": {"nu": float(nu), "lambda": float(lambd)},
        },
        "bootstrap": bootstrap_config_dict,
        "optimizer": optimization_config_dict,
    }

    config_id = _config_id(config_data)
    os.makedirs(f"runs/{config_dir}/configs", exist_ok=True)
    with open(f"runs/{config_dir}/configs/{config_id}.yaml", "w") as outfile:
        yaml.dump(config_data, outfile, default_flow_style=False)

    return config_id


def run_bootstrap_from_config(
    config_filename, config_dir, verbose=True, check_if_exists_already=True
):

    # optionally skip if data file already exists
    if check_if_exists_already:
        logger.info(f"runs/{config_dir}/results/{config_filename}.json")
        if os.path.exists(f"runs/{config_dir}/results/{config_filename}.json"):
            logger.info("Run result already exists, skipping.")
            return

    # load the config file
    with open(f"runs/{config_dir}/configs/{config_filename}.yaml") as stream:
        config = yaml.safe_load(stream)
    config_model = config["model"]
    config_bootstrap = config["bootstrap"]
    config_optimizer = config["optimizer"]

    # build the model
    model = _MODEL_CLASSES[config_model["model name"]](
        couplings=config_model["couplings"]
    )

    # cache paths
    structural_cache_path = f"cache/structural/{_struct_hash(config)}"
    config_cache_path = f"cache/per_config/{config_filename}"

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
        st_operator_to_minimize = model.operators_to_track[
            config_bootstrap["st_operator_to_minimize"]
        ]

    # operators whose expectation values are to be fixed
    st_operator_inhomo_constraints = [(SingleTraceOperator(data={(): 1}), 1)]
    if config_bootstrap["st_operators_evs_to_set"] is not None:
        for key, value in config_bootstrap["st_operators_evs_to_set"].items():
            st_operator_inhomo_constraints.append(
                (model.operators_to_track[key], value)
            )

    # build the bootstrap
    bootstrap = _BOOTSTRAP_CLASSES[config_model["bootstrap class"]](
        matrix_system=model.matrix_system,
        hamiltonian=model.hamiltonian,
        gauge_generator=model.gauge_generator,
        max_degree_L=config_bootstrap["max_degree_L"],
        odd_degree_vanish=config_bootstrap["odd_degree_vanish"],
        simplify_quadratic=config_bootstrap["simplify_quadratic"],
        symmetry_generators=model.symmetry_generators,
        structural_cache_path=structural_cache_path,
        config_cache_path=config_cache_path,
    )

    # load from cache: structural cache is always used; per-config cache is
    # used when load_from_previously_computed is set
    bootstrap.load_structural_cache(structural_cache_path)
    if config_bootstrap["load_from_previously_computed"]:
        bootstrap.load_config_cache(config_cache_path)

    # solve the bootstrap
    optimization_method = config_optimizer.pop("optimization_method")
    if optimization_method == "newton":
        param, optimization_result = solve_bootstrap_newton(
            bootstrap=bootstrap,
            st_operator_to_minimize=st_operator_to_minimize,
            st_operator_inhomo_constraints=st_operator_inhomo_constraints,
            **config_optimizer,
        )
    elif optimization_method == "pytorch":
        param, optimization_result = solve_bootstrap_pytorch(
            bootstrap=bootstrap,
            st_operator_to_minimize=st_operator_to_minimize,
            st_operator_inhomo_constraints=st_operator_inhomo_constraints,
            **config_optimizer,
        )

    if param is None:
        return

    # record select expectation values
    expectation_values = {
        name: float(
            bootstrap.get_operator_expectation_value(
                st_operator=st_operator, param=param
            ).real
        )
        for name, st_operator in model.operators_to_track.items()
    }

    # save the results
    result = optimization_result | expectation_values
    result["param"] = list(param)

    os.makedirs(f"runs/{config_dir}/results", exist_ok=True)
    with open(f"runs/{config_dir}/results/{config_filename}.json", "w") as f:
        json.dump(result, f)

    for key, value in expectation_values.items():
        logger.info(f"EV {key}: {value:.4e}")

    return result


def _init_worker_logging():
    import warnings

    logging.basicConfig(level=logging.INFO)
    # In parallel workers, suppress per-step detail logs — interleaved output
    # from multiple workers is unreadable. Only results and errors surface.
    logging.getLogger("matrixbootstrap.bootstrap").setLevel(logging.WARNING)
    logging.getLogger("matrixbootstrap.algebra").setLevel(logging.ERROR)
    logging.getLogger("matrixbootstrap.solver_newton").setLevel(logging.WARNING)
    # Suppress cvxpy's "Solution may be inaccurate" UserWarning in workers.
    warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")


def _build_config_cache(config_filename, config_dir):
    """
    Pre-build and save all coupling-dependent cache artifacts for a single config.
    Subsequent runs will load from cache instead of recomputing.
    """
    with open(f"runs/{config_dir}/configs/{config_filename}.yaml") as stream:
        config = yaml.safe_load(stream)
    config_model = config["model"]
    config_bootstrap = config["bootstrap"]

    structural_cache_path = f"cache/structural/{_struct_hash(config)}"
    config_cache_path = f"cache/per_config/{config_filename}"

    # skip if already fully built
    if os.path.exists(config_cache_path + "/bootstrap_table_sparse.npz"):
        logger.info("Config cache already complete: %s", config_cache_path)
        return

    logger.info("Building config cache: %s", config_cache_path)

    model = _MODEL_CLASSES[config_model["model name"]](
        couplings=config_model["couplings"]
    )
    if not config_bootstrap["impose_symmetries"]:
        model.symmetry_generators = None
    if not config_bootstrap["impose_gauge_symmetry"]:
        model.gauge_generator = None

    bootstrap = _BOOTSTRAP_CLASSES[config_model["bootstrap class"]](
        matrix_system=model.matrix_system,
        hamiltonian=model.hamiltonian,
        gauge_generator=model.gauge_generator,
        max_degree_L=config_bootstrap["max_degree_L"],
        odd_degree_vanish=config_bootstrap["odd_degree_vanish"],
        simplify_quadratic=config_bootstrap["simplify_quadratic"],
        symmetry_generators=model.symmetry_generators,
        structural_cache_path=structural_cache_path,
        config_cache_path=config_cache_path,
    )

    # load structural cache (cyclic constraints) if available
    bootstrap.load_structural_cache(structural_cache_path)

    # build and save all components in dependency order
    bootstrap.build_null_space_matrix()  # → build_linear_constraints → generate_constraints
    bootstrap.build_quadratic_constraints()  # needs null_space_matrix
    bootstrap.build_bootstrap_table()  # needs null_space_matrix


def _build_all_caches(config_filenames, config_dir):
    """
    Pre-build per-config caches for all configs in a run.
    Each config has its own cache keyed by config_id.
    """
    for config_filename in config_filenames:
        _build_config_cache(config_filename, config_dir)


def run_all_configs(
    config_dir,
    parallel=False,
    max_workers=6,
    verbose=True,
    check_if_exists_already=True,
):

    config_filenames = os.listdir(f"runs/{config_dir}/configs")
    config_filenames = [f[:-5] for f in config_filenames if ".yaml" in f]
    # np.random.shuffle(config_filenames) # shuffle

    if not parallel:
        for config_filename in config_filenames:
            run_bootstrap_from_config(
                config_filename,
                config_dir,
                check_if_exists_already=check_if_exists_already,
            )
    else:
        # Pre-build structural caches serially to avoid parallel write races.
        # Each unique (model, L) maps to one shared cache; building one config
        # is sufficient since _build_config_cache skips if already complete.
        seen_struct_hashes = set()
        for config_filename in config_filenames:
            with open(f"runs/{config_dir}/configs/{config_filename}.yaml") as f:
                import yaml as _yaml

                _cfg = _yaml.safe_load(f)
            sh = _struct_hash(_cfg)
            if sh not in seen_struct_hashes:
                seen_struct_hashes.add(sh)
                struct_path = f"cache/structural/{sh}"
                if not os.path.exists(struct_path):
                    logger.info("Pre-building structural cache: %s", struct_path)
                    _build_config_cache(config_filename, config_dir)

        with ProcessPoolExecutor(
            max_workers, initializer=_init_worker_logging
        ) as executor:
            futures = [
                executor.submit(
                    run_bootstrap_from_config,
                    config_filename,
                    config_dir,
                    verbose,
                    check_if_exists_already,
                )
                for config_filename in config_filenames
            ]
        for future in futures:
            future.result()
        logger.info("finished!")


if __name__ == "__main__":
    fire.Fire()
