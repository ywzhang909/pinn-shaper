# Physics-Informed Neural Networks for Optimal Beam Shaping in Flat Optics

![N|Solid](/setup.png)

Pinn-shaper provides research code and example notebooks for designing high-fidelity and high-performance phase profiles that transform a known input beam into a prescribed target intensity.

Compared with Gerchberg--Saxton and direct gradient-based optimization methods, the PINN produces smoother phase profiles with improved robustness, less speckle, and can direct more power into the desired target region. It enforces the underlying beam-shaping physics throughout training, which helps guide the solution toward physically meaningful phase maps.

The method supports both finite-distance and far-field beam shaping, including non-paraxial configurations. The resulting phases can be displayed on spatial light modulators (SLMs) or used as starting points for metasurface inverse design and topology optimization. The PINN is specially simple to train for low-spatial-frequency target patterns.

## Requirements

- `torch`
- `diffractsim`

For a CUDA-enabled installation, install the appropriate PyTorch build for your operating system.

## Main features

- Finite-distance phase design with `Nearfield_PINN`.
- Far-field angular phase design with `Farfield_PINN`.
- Non-paraxial generalized-Snell-law ray mappings.
- Energy conservation enforced through Monge--Ampère PDE residuals.
- Automatic differentiation of the first- and second-order phase derivatives.
- Automatic power normalization between the source and target distributions.
- Optional symmetry reduction for symmetric optical designs.
- Scalar diffraction validation with `diffractsim`.
- Gerchberg--Saxton comparison for the far-field example.
- Example trained models and target images for reproducing the supplied results.

## Citation

If you use `pinn-shaper` in your research, please cite the paper:

R. de la Fuente Herrezuelo, <i>Physics-Informed Neural Networks for Optimal Beam Shaping in Flat Optics</i> (2026), https://doi.org/10.48550/arXiv.2607.18012.