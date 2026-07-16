# pinn-shaper

Physics-Informed Neural Networks for optimal beam shaping in flat optics.

This repository contains the code and notebooks used to design phase-only masks with PINNs and to compare the far-field result with the Gerchberg--Saxton algorithm.

## Requirements

Recommended:

- Python 3.10 or newer
- `pip`

Main Python packages:

- `torch`
- `numpy`
- `scipy`
- `matplotlib`
- `pillow`
- `jax` and `jaxlib`
- `diffractsim`
- `jupyter` or `notebook`

## Installation

Install the required packages:

```bash
python -m pip install --upgrade pip
pip install torch numpy scipy matplotlib pillow jax jaxlib diffractsim jupyter
```

If you need a CUDA-enabled version of PyTorch, install `torch` using the command recommended for your system on the official PyTorch installation page, then install the remaining packages with `pip`.

