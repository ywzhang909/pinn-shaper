# Physics-Informed Neural Networks for Optimal Beam Shaping in Flat Optics
[![DOI](https://zenodo.org/badge/DOI/10.48550/arXiv.2607.18012.svg)](https://doi.org/10.48550/arXiv.2607.18012)

![N|Solid](images/setup.png)

PINN-shaper provides research code and example notebooks for designing high-fidelity and high-performance phase profiles that transform a known input beam into a prescribed target intensity.

Compared with Gerchberg--Saxton and direct gradient-based optimization methods, this work reformulates the beam-shaping problem as a nonlinear PDE, which is solved with a PINN whose solutions yield smoother phase profiles, improved robustness, reduced speckle, and greater power directed to the desired target region. It enforces the underlying beam-shaping physics throughout training, which helps guide the solution toward physically meaningful phase maps.

The method supports both finite-distance and far-field beam shaping, including non-paraxial configurations. The resulting phases can be displayed on spatial light modulators (SLMs) or used as optimal starting points for further metasurface inverse design and topology optimization. The PINN is specially simple to train for low-spatial-frequency target patterns.

|<img src="https://github.com/rafael-fuente/pinn-shaper/blob/main/images/star_flat_top_near_field_propagation.gif" width="100%">|<img src="https://github.com/rafael-fuente/pinn-shaper/blob/main/images/dt_letters_flat_top_near_field_propagation.gif" width="95%">|
:--------------------:|:--------------------:|
`Star-shaped flat-top (near field).ipynb` | `DT-letters-shaped flat-top (near field).ipynb` |
[Link to the example](https://github.com/rafael-fuente/pinn-shaper/blob/main/Star-shaped%20flat-top%20(near%20field).ipynb)| [Link to the example](https://github.com/rafael-fuente/pinn-shaper/blob/main/DT-letters-shaped%20flat-top%20(near%20field).ipynb)|



## Requirements

For installation, just clone this repository. The required packages are:

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

## Reproduction results

This repository also ships a from-scratch reproduction of the four paper
experiments (trained locally, not the supplied `saved_models/`), validated
against the paper's metrics with `diffractsim` on a CPU backend. Full details
and figures for every experiment are in [`report.md`](report.md).

| Experiment | Task | Target plane | Metric |
|---|---|---|---|
| Exp1 | Star flat-top | near field, z = 10 cm | cross-section rel-RMS 2.45% |
| Exp2 | Star flat-top | far field | cross-section rel-RMS 0.76% |
| Exp3 | DT letters | near field (6 → 10 mm) | cross-section rel-RMS 5.38% |
| Exp4 | DT letters + GS | far field | PINN MSE 4.03×10⁶ vs GS 2.08×10⁸ |

The paper's headline comparison is reproduced: on the far-field DT-letters
target, the PINN reaches MSE ≈ 4.0×10⁶ (paper 3.59×10⁶) versus Gerchberg–Saxton
2.08×10⁸ (paper 2.08×10⁸, matched to the digit), i.e. a ~52× MSE improvement,
with 99.96% (PINN) vs 95.90% (GS) target-area power efficiency. Using the
supplied official model, the pipeline reproduces the paper numbers exactly
(MSE 3.590469×10⁶, rel-RMS 2.97%, η 99.982% / 95.898%).

Reproduction artifacts live in `repro/` (scripts, `--seed` / `--no-train`),
`reproduced_models/` (four from-scratch `.pth`), `results/` (figures) and
`report.md` (report). Training used the residual GPU memory of two RTX 4090s
(~5 GB free per card); each experiment takes 3–5 minutes.

## Citation

If you use `pinn-shaper` in your research, please cite the paper:

R. de la Fuente Herrezuelo, <i>Physics-Informed Neural Networks for Optimal Beam Shaping in Flat Optics</i> (2026), https://doi.org/10.48550/arXiv.2607.18012.