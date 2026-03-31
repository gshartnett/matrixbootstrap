"""
L=3 mini-BMN invariant-basis run using CLARABEL with increased static regularization.

Generates CLARABEL configs for all 7 nu values and runs them sequentially.
Based on Attempt 18: static_reg=1e-4 prevents CLARABEL NumericalError that
occurred with the default 1e-7 setting.

Usage:
    conda run -n bmnsim python scripts/run_l3_clarabel.py
    conda run -n bmnsim python scripts/run_l3_clarabel.py --max_iter 100 --static_reg 1e-4
    conda run -n bmnsim python scripts/run_l3_clarabel.py --nu 3.0  # single nu value
"""

import argparse
import json
import logging
import os
import sys
import time

import yaml

LOG_FILE = "runs/l3_clarabel.log"
os.makedirs("runs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--max_iter", type=int, default=50, help="CLARABEL max_iter")
parser.add_argument(
    "--static_reg",
    type=float,
    default=1e-4,
    help="CLARABEL static_regularization_constant",
)
parser.add_argument(
    "--nu",
    type=float,
    default=None,
    help="Run only this nu value (default: all 7)",
)
args = parser.parse_args()

CONFIG_DIR = "MiniBMN_invariant_L3_clarabel"
SOURCE_CONFIG_DIR = "MiniBMN_invariant_L3"
os.makedirs(f"runs/{CONFIG_DIR}/configs", exist_ok=True)
os.makedirs(f"runs/{CONFIG_DIR}/results", exist_ok=True)

# Load source configs and generate CLARABEL versions.
source_configs = sorted(os.listdir(f"runs/{SOURCE_CONFIG_DIR}/configs"))
configs_to_run = []
for fname in source_configs:
    if not fname.endswith(".yaml"):
        continue
    with open(f"runs/{SOURCE_CONFIG_DIR}/configs/{fname}") as f:
        config = yaml.safe_load(f)

    nu = config["model"]["couplings"]["nu"]
    if args.nu is not None and abs(nu - args.nu) > 1e-9:
        continue

    # Patch optimizer to use CLARABEL with increased static_reg.
    config["optimizer"]["cvxpy_solver"] = "CLARABEL"
    config["optimizer"]["maxiters_cvxpy"] = args.max_iter
    config["optimizer"]["clarabel_static_reg"] = args.static_reg
    # maxiters=1 is fine (linear-only problem exits after step 1 anyway).
    config["optimizer"]["maxiters"] = 1

    # Use the same config_id as the source (nu determines the physics, not solver).
    config_id = fname[:-5]  # strip .yaml
    out_path = f"runs/{CONFIG_DIR}/configs/{config_id}.yaml"
    with open(out_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    configs_to_run.append((config_id, nu))
    logger.info("Prepared config %s (nu=%.1f)", config_id, nu)

logger.info(
    "Starting L=3 CLARABEL run: %d configs, max_iter=%d, static_reg=%.1e",
    len(configs_to_run),
    args.max_iter,
    args.static_reg,
)

from matrixbootstrap.config_utils import run_bootstrap_from_config  # noqa: E402

total_t0 = time.time()
for config_id, nu in configs_to_run:
    result_path = f"runs/{CONFIG_DIR}/results/{config_id}.json"
    if os.path.exists(result_path):
        logger.info("Config %s (nu=%.1f) already done, skipping.", config_id, nu)
        continue

    logger.info("=== Running nu=%.1f (config %s) ===", nu, config_id)
    t0 = time.time()
    try:
        run_bootstrap_from_config(config_id, CONFIG_DIR, check_if_exists_already=False)
    except Exception as e:
        logger.error("Config %s (nu=%.1f) FAILED: %s", config_id, nu, e)
    elapsed = time.time() - t0
    logger.info("Config %s done in %.1f min", config_id, elapsed / 60)

    # Print summary if result exists.
    if os.path.exists(result_path):
        with open(result_path) as f:
            result = json.load(f)
        min_eig = result.get("min_bootstrap_eigenvalue", float("nan"))
        energy = result.get("energy", result.get("prob.value", float("nan")))
        status = result.get("prob.status", "?")
        logger.info(
            "  nu=%.1f: E_bound=%.6f, min_eig=%.4e, status=%s",
            nu,
            energy,
            min_eig,
            status,
        )
    else:
        logger.warning("  nu=%.1f: no result file (run failed).", nu)

total_elapsed = time.time() - total_t0
logger.info("All done. Total wall time: %.1f min", total_elapsed / 60)
