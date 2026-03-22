# matrixbootstrap

A Python implementation of the **matrix bootstrap** method for quantum mechanical matrix models. Given a matrix model Hamiltonian, the code uses semidefinite programming (SDP) to compute rigorous bounds on large-N ground state energies and operator expectation values.

## Background

The matrix bootstrap is an analytic/numerical method that constrains the expectation values of single-trace operators — quantities like $\langle \tfrac{1}{N} \text{tr}(X^2) \rangle$ — without solving the Schrödinger equation directly. Instead, it enforces three consistency conditions that any physical state must satisfy:

1. **Gauge invariance**: expectation values must be invariant under the U(N) gauge symmetry of the model.
2. **Hermiticity**: the bootstrap matrix $M_{ij} = \langle O_i^\dagger O_j \rangle$ must be positive semidefinite.
3. **Equations of motion**: $\langle [H, O] \rangle = 0$ for all operators $O$.

These conditions translate into linear constraints on a finite-dimensional vector of expectation values, plus a semidefinite constraint on the bootstrap matrix. The SDP then either finds a feasible point (a valid expectation value vector) or proves infeasibility (ruling out a region of coupling space).

The truncation parameter $L$ controls which operators are included: all single-trace monomials up to degree $2L$ in the fundamental fields $(X_i, \Pi_i)$. Larger $L$ gives tighter bounds at the cost of a larger SDP.

## Supported models

All models are large-N matrix quantum mechanics (0+1d), working in the gauge $A_0 = 0$.

| Class | Description | Couplings |
|---|---|---|
| `OneMatrix` | Single-matrix anharmonic oscillator | `g2`, `g4`, `g6` |
| `TwoMatrix` | Two-matrix model with commutator-squared interaction | `g2`, `g4` |
| `ThreeMatrix` | Three-matrix model | `g2`, `g3`, `g4` |
| `MiniBFSS` | Mini-BFSS: massless three-matrix model ($g_2=g_3=0$) | `lambda` |
| `MiniBMN` | Mini-BMN: three-matrix model with mass and cubic terms | `nu`, `lambda` |

`MiniBFSS` and `MiniBMN` are bosonic truncations of the BFSS and BMN matrix models, which arise as dimensionally reduced descriptions of M-theory.

## Installation

### Prerequisites

- Python 3.10–3.12
- [conda](https://docs.conda.io/en/latest/miniconda.html)
- [Poetry](https://python-poetry.org/docs/#installation)
- [Homebrew](https://brew.sh) (macOS only)

### Steps

**1. Clone the repo**

```bash
git clone git@github.com:gshartnett/matrixbootstrap.git
cd matrixbootstrap
```

**2. Install SuiteSparse**

`matrixbootstrap` uses [PySPQR](https://github.com/yig/PySPQR) (`sparseqr`) for sparse null space computation via SuiteSparse SPQR. SuiteSparse must be installed before `sparseqr` can compile its CFFI bindings.

```bash
# macOS (Homebrew)
brew install suite-sparse

# Ubuntu/Debian
sudo apt-get install libsuitesparse-dev

# conda
conda install -c conda-forge suitesparse
```

**3. Create and activate a conda environment**

```bash
conda create -n matrixbootstrap python=3.12
conda activate matrixbootstrap
```

**4. Install the package**

```bash
poetry install
```

This installs `matrixbootstrap` and all dependencies (numpy, scipy, cvxpy, torch, sparseqr, etc.) into the active environment.

### Troubleshooting: sparseqr on macOS 15 (Sequoia)

On macOS 15, importing `sparseqr` may crash the Python process with `exit code 137` (SIGKILL) if the conda-bundled `libgfortran` or `libquadmath` libraries were modified by `install_name_tool` without subsequent re-signing. macOS 15 enforces code signature validity and kills any process loading a tampered library.

**Symptom**: `import sparseqr` causes a silent SIGKILL with no traceback.

**Diagnosis**: check whether the libraries have valid signatures:
```bash
codesign -v $CONDA_PREFIX/lib/libgfortran.5.dylib
codesign -v $CONDA_PREFIX/lib/libquadmath.0.dylib
# "invalid signature" means the library has been tampered with
```

**Fix**: re-sign the affected libraries with an ad-hoc signature:
```bash
codesign --force -s - $CONDA_PREFIX/lib/libgfortran.5.dylib
codesign --force -s - $CONDA_PREFIX/lib/libquadmath.0.dylib
```

After re-signing, `import sparseqr` should succeed.

## Usage

### Quickstart: one-matrix model

```python
from matrixbootstrap.models import OneMatrix
from matrixbootstrap.bootstrap import BootstrapSystem

# Define the model: harmonic oscillator (g4=g6=0)
model = OneMatrix(couplings={"g2": 1.0, "g4": 0.0, "g6": 0.0})

# Build the bootstrap system at truncation level L=2
bootstrap = BootstrapSystem(
    matrix_system=model.matrix_system,
    hamiltonian=model.hamiltonian,
    gauge_generator=model.gauge_generator,
    max_degree_L=2,
    odd_degree_vanish=True,   # parity symmetry sets odd-degree expectation values to zero
    simplify_quadratic=True,
    checkpoint_path="checkpoints/one_matrix_L2",
)

# Build linear constraints (gauge invariance + equations of motion)
bootstrap.build_linear_constraints()

# Compute the null space: all expectation value vectors satisfying the linear constraints
bootstrap.build_null_space_matrix()
```

The null space matrix `bootstrap.null_space_matrix` parametrises all physically admissible expectation value vectors. The SDP then searches this space for a positive-semidefinite bootstrap matrix.

### Running the SDP

The Newton-step outer loop solver (`solver_newton.py`) iteratively solves a sequence of SDPs to find the ground state energy. The recommended entry point is the batch runner scripts in `scripts/`:

```bash
# Run the one-matrix model sweep
python scripts/one_matrix.py

# Run mini-BFSS
python scripts/mini_bfss.py
```

These generate config files and execute them in parallel. Results are saved as pickle files and can be plotted with:

```bash
python scripts/plotter.py
```

### Mini-BMN example

The mini-BMN model has two couplings: the dimensionless mass parameter $\nu$ and the 't Hooft coupling $\lambda$. It has a known supersymmetric ground state at $E=0$ when $\nu^2 = \lambda$, which provides a useful check:

```python
from matrixbootstrap.models import MiniBMN

# At nu^2 = lambda, the ground state energy should be zero (SUSY ground state)
model = MiniBMN(couplings={"nu": 1.0, "lambda": 1.0})
```

## Repository structure

```
matrixbootstrap/
├── algebra.py             # SingleTraceOperator, MatrixSystem, commutation rules
├── bootstrap.py           # BootstrapSystem: constraint building and null space
├── models.py              # OneMatrix, TwoMatrix, ThreeMatrix, MiniBFSS, MiniBMN
├── solver_newton.py       # Newton outer loop SDP solver (main solver)
├── solver_trustregion.py  # Trust-region SDP solver (alternative)
├── solver_pytorch.py      # PyTorch-based gradient descent solver (experimental)
├── linear_algebra.py      # Sparse null/row space via sparseqr (SuiteSparse SPQR)
├── group_theory.py        # SU(N) generators, SU(2) irreps
├── brezin.py              # Large-N analytic solution (Brezin et al. 1978)
├── born_oppenheimer.py    # Born-Oppenheimer approximation utilities
├── config_utils.py        # Config file generation and parallel execution
└── utils.py               # Miscellaneous utilities

scripts/
├── one_matrix.py          # Batch runner: one-matrix model
├── two_matrix.py          # Batch runner: two-matrix model
├── mini_bfss.py           # Batch runner: mini-BFSS
├── mini_bmn.py            # Batch runner: mini-BMN
└── plotter.py             # Result plotting

tests/
├── test_models.py         # Hamiltonian Hermiticity; [H, generator] = 0
├── test_bootstrap.py      # Null space satisfies linear constraints
├── test_brezin.py         # Reproduces Brezin et al. (1978) Table 3
├── test_group_theory.py   # SU(2) Casimir; SU(N) Lie algebra
└── test_linear_algebra.py # Coefficient extraction utilities
```

## Running tests

```bash
conda activate matrixbootstrap
pytest tests/ -v
```

## References

- Han, Hartnoll, Kruthoff, [*Bootstrapping Matrix Quantum Mechanics*](https://arxiv.org/abs/2004.01981), PRL 2020
- Brezin, Itzykson, Parisi, Zuber, [*Planar diagrams*](https://link.springer.com/article/10.1007/BF01208266), Commun. Math. Phys. 59 (1978)
- Lin, [*Bootstrapping the Minimal BFSS*](https://arxiv.org/abs/2310.01703), 2023
