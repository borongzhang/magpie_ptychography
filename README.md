# MAGPIE: A Multilevel-Adaptive-Guided Solver for Ptychographic Phase Retrieval

This repository contains the reference implementation for the paper **"MAGPIE: A Multilevel-Adaptive-Guided Solver for Ptychographic Phase Retrieval"** ([arXiv:2504.10118](https://arxiv.org/abs/2504.10118)).

Authors:
- [Borong Zhang](https://borongzhang.com/)
- [Qin Li](https://sites.google.com/view/qinlimadison/home)
- [Zichao (Wendy) Di](https://www.anl.gov/profile/zichao-di)

MAGPIE is implemented in Python and uses NumPy/SciPy for computation and Jupyter notebooks for reproducible experiments and visualization.

## What this repository includes

- Core MAGPIE solver implementation.
- Numerical experiments from the paper.
- Scripts and notebooks to reproduce reported figures and results.

## Installation

### Option 1: Install from GitHub (recommended)

```bash
conda create -n magpie-env python=3.11 -y
conda activate magpie-env
pip install git+https://github.com/borongzhang/magpie_ptychography.git@main
```

### Option 2: Local editable install (for development)

```bash
git clone https://github.com/borongzhang/magpie_ptychography.git
cd magpie_ptychography
conda create -n magpie-env python=3.11 -y
conda activate magpie-env
pip install -e .
```

### Optional: register a Jupyter kernel

```bash
python -m ipykernel install --user --name magpie-env --display-name "Python (magpie-env)"
```

## Reproducibility

All numerical examples in the manuscript are included in this repository. To reproduce results:

1. Create and activate the environment above.
2. Run the provided experiment scripts and/or open the notebooks.
3. Compare generated outputs with figures/tables in the paper.

> If you use MAGPIE in research, please cite the paper linked above.

## Troubleshooting

- **Kernel not visible in Jupyter**: run the `ipykernel install` command again, then restart Jupyter.
- **Dependency issues**: ensure Python 3.11 is active in `magpie-env` before installation.
- **Slow first run**: initial setup and data generation may take longer due to environment warm-up.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
