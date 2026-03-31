"""
Defensive L=3 mini-BMN invariant-basis run.

Safeguards:
  - Single worker (no parallelism) to bound peak memory
  - Memory cap via resource.setrlimit (kills process gracefully before OOM crash)
  - All output logged to runs/l3_invariant_safe.log
  - Processes one config at a time; already-done configs are skipped

Usage:
    conda run -n bmnsim python scripts/run_l3_invariant_safe.py
    conda run -n bmnsim python scripts/run_l3_invariant_safe.py --mem_gb 12
"""

import argparse
import logging
import os
import resource
import sys
import time

from matrixbootstrap.config_utils import run_all_configs

# ── logging ────────────────────────────────────────────────────────────────
LOG_FILE = "runs/l3_invariant_safe.log"
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

# ── argument parsing ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mem_gb",
    type=float,
    default=10.0,
    help="Hard memory limit in GB (process killed if exceeded)",
)
args = parser.parse_args()

# ── memory cap ─────────────────────────────────────────────────────────────
mem_bytes = int(args.mem_gb * 1024**3)
try:
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    logger.info("Memory cap set to %.1f GB", args.mem_gb)
except (ValueError, resource.error) as e:
    logger.warning("Could not set memory cap: %s", e)

# ── run configs ──────────────────────────────────────────────────────────────

CONFIG_DIR = "MiniBMN_invariant_L3"

logger.info(
    "Starting L=3 invariant basis run (single worker, mem_cap=%.1f GB)", args.mem_gb
)
t0 = time.time()

run_all_configs(
    config_dir=CONFIG_DIR,
    parallel=False,  # single worker — safe
    max_workers=1,
    check_if_exists_already=True,
)

elapsed = time.time() - t0
logger.info("Done. Total wall time: %.1f min", elapsed / 60)
