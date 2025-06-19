# BMN Simulation Package

A Python package for simulating and analyzing BMN (Berenstein-Maldacena-Nastase) matrix models using bootstrap methods and convex optimization.

## Features

- Bootstrap analysis for matrix models
- Support for one-matrix and two-matrix models
- BFSS and BMN model implementations
- Convex optimization using CVXPY and PyTorch
- Visualization and plotting utilities
- Jupyter notebook integration

## Installation

### Prerequisites

- Python 3.10 or greater
- Poetry (for dependency management)

### Install with Poetry

1. Clone the repository:
```bash
git clone <repository-url>
cd bmnsim
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Activate the virtual environment:
```bash
poetry shell
```

### Manual Installation (Alternative)

If you prefer pip:
```bash
pip install numpy pandas scipy matplotlib jupyter cvxpy torch fire cycler
```

### Special Note for M2 Mac Users

To install sparseqr on M2 Macbook Air:
- Manually copy files from the `sparseqr` directory of https://github.com/yig/PySPQR
- If you encounter library issues, follow these steps:
  - `conda install openblas`
  - `brew install lapack`
  - `pip uninstall numpy`
  - `pip install numpy==1.21.1`
  - `conda update --all`

## Basic Usage

### One Matrix Model

```python
from bmn.config_utils import generate_config_one_matrix, run_all_configs
from bmn.brezin import compute_Brezin_energy

# Generate configuration for one matrix model
generate_config_one_matrix(
    config_filename="example",
    config_dir="OneMatrix_L_3_example",
    g2=1.0,
    g4=1.0,
    g6=0.0,
    max_degree_L=3,
    impose_symmetries=True,
    optimization_method="cvxpy"
)

# Run the optimization
run_all_configs(
    config_dir="OneMatrix_L_3_example",
    parallel=False
)

# Compare with analytic result
energy = compute_Brezin_energy(g_value=0.25)
print(f"Brezin energy: {energy}")
```

### Loading and Analyzing Results

```python
import pandas as pd
from bmn.plotting_utils import load_data

# Load simulation results
df = load_data(
    datadir="data/OneMatrix_L_3",
    names_in_filename=["g4_"],
    tol=1e-6
)

# Analyze results
print(f"Number of data points: {len(df)}")
print(f"Energy range: {df['energy'].min():.4f} - {df['energy'].max():.4f}")
```

## Project Structure

```
bmnsim/
├── bmn/                    # Main package
│   ├── bootstrap.py        # Bootstrap analysis tools
│   ├── brezin.py          # Brezin energy calculations
│   ├── config_utils.py    # Configuration utilities
│   └── ...
├── data/                  # Simulation data
├── scripts/               # Analysis scripts
├── notebooks/             # Jupyter notebooks
└── tests/                 # Test files
```

## Running Tests

```bash
# Run all tests
poetry run pytest

# Run specific test
poetry run pytest tests/test_one_matrix.py

# Run with verbose output
poetry run pytest -v
```

## Development

### Code Formatting

```bash
# Format code with black
poetry run black .

# Sort imports with isort
poetry run isort .
```

### Adding Dependencies

```bash
# Add runtime dependency
poetry add package-name

# Add development dependency
poetry add --group dev package-name
```

## Examples

See the `notebooks/` directory for detailed examples:
- `Plotting Notebook.ipynb` - Visualization and analysis examples
- Example configurations in `scripts/tests/`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

[Add your license information here]

## Citation

If you use this package in your research, please cite:
[Add citation information here]