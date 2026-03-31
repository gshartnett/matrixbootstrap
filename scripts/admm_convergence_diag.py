"""
ADMM convergence diagnostic — runs a single L=3 config for a limited number
of iterations with frequent logging to assess whether the custom ADMM-BLAS
solver is making meaningful progress toward PSD feasibility.

Usage:
    conda run -n bmnsim python scripts/admm_convergence_diag.py
    conda run -n bmnsim python scripts/admm_convergence_diag.py --maxiters 20000 --log_interval 200
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import yaml

LOG_FILE = "runs/admm_convergence_diag.log"
os.makedirs("runs/admm_convergence_diag", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),  # overwrite each run
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--maxiters", type=int, default=10000)
parser.add_argument("--log_interval", type=int, default=100)
parser.add_argument("--rho", type=float, default=1.0)
parser.add_argument("--alpha", type=float, default=1.5)
parser.add_argument(
    "--config",
    default="runs/MiniBMN_invariant_L3/configs/04fab7db5301.yaml",  # nu=3.0
)
args = parser.parse_args()

logger.info(
    "ADMM convergence diagnostic: maxiters=%d, log_interval=%d, rho=%.3g, alpha=%.3g",
    args.maxiters,
    args.log_interval,
    args.rho,
    args.alpha,
)
logger.info("Config: %s", args.config)

# Load and patch the config, then drive the solver directly.
with open(args.config) as f:
    config = yaml.safe_load(f)

config["optimizer"]["maxiters_cvxpy"] = args.maxiters
config["optimizer"]["admm_rho"] = args.rho
config["optimizer"]["admm_alpha"] = args.alpha
config["optimizer"]["admm_log_interval"] = args.log_interval
config["optimizer"]["maxiters"] = 1  # one Newton step (linear-only, exits after step 1)

from matrixbootstrap.algebra import SingleTraceOperator  # noqa: E402
from matrixbootstrap.config_utils import (  # noqa: E402
    _BOOTSTRAP_CLASSES,
    _MODEL_CLASSES,
    _struct_hash,
)
from matrixbootstrap.solver_newton import (  # noqa: E402
    solve_bootstrap as solve_bootstrap_newton,
)

config_model = config["model"]
config_bootstrap = config["bootstrap"]
config_optimizer = config["optimizer"]

model = _MODEL_CLASSES[config_model["model name"]](couplings=config_model["couplings"])

structural_cache_path = f"cache/structural/{_struct_hash(config)}"
config_id = os.path.splitext(os.path.basename(args.config))[0]
config_cache_path = f"cache/per_config/{config_id}"

if not config_bootstrap["impose_symmetries"] and not config_bootstrap.get(
    "use_invariant_basis", False
):
    model.symmetry_generators = None
if not config_bootstrap["impose_gauge_symmetry"]:
    model.gauge_generator = None

st_operator_to_minimize = model.operators_to_track[
    config_bootstrap["st_operator_to_minimize"]
]
st_operator_inhomo_constraints = [(SingleTraceOperator(data={(): 1}), 1)]

bootstrap_class = _BOOTSTRAP_CLASSES[config_model["bootstrap class"]]
bootstrap = bootstrap_class(
    matrix_system=model.matrix_system,
    hamiltonian=model.hamiltonian,
    gauge_generator=model.gauge_generator,
    max_degree_L=config_bootstrap["max_degree_L"],
    odd_degree_vanish=config_bootstrap["odd_degree_vanish"],
    simplify_quadratic=config_bootstrap["simplify_quadratic"],
    symmetry_generators=model.symmetry_generators,
    structural_cache_path=structural_cache_path,
    config_cache_path=config_cache_path,
    use_invariant_basis=config_bootstrap.get("use_invariant_basis", False),
)
if config_bootstrap.get("use_invariant_basis", False):
    st_operator_to_minimize = bootstrap.hamiltonian
bootstrap.load_structural_cache(structural_cache_path)
bootstrap.load_config_cache(config_cache_path)

optimization_method = config_optimizer.pop("optimization_method")
t0 = time.time()
param, result = solve_bootstrap_newton(
    bootstrap=bootstrap,
    st_operator_to_minimize=st_operator_to_minimize,
    st_operator_inhomo_constraints=st_operator_inhomo_constraints,
    **config_optimizer,
)
elapsed = time.time() - t0

logger.info("=== DIAGNOSTIC RESULT ===")
logger.info("Elapsed: %.1f min", elapsed / 60)
if result:
    for k, v in result.items():
        if not isinstance(v, (list, np.ndarray)):
            logger.info("  %s: %s", k, v)
