# QSP Phase Factorization via Logarithmic Derivatives

Reference implementation for the paper *QSP Phase Factorization via Logarithmic Derivatives* (Christopher Peng, 2026).

Given a real target function on [-1, 1] with definite parity, the code computes quantum signal processing (QSP) phase factors in O(d log² d) time. It follows the stable factorization pipeline of Ying (2022) but replaces Prony's method: the Fourier coefficients of the logarithmic derivative h′/h give the power sums of the roots inside the unit disk, and the Newton–Girard identities (computed as a power-series exponential) recover the characteristic polynomial without root finding.

## Requirements

- Python 3
- `numpy`
- `scipy`
- `matplotlib`
- `qsppack` (the Python version of QSPPACK, used for the Chebyshev-to-Laurent conversion and for verifying the phase factors)

```bash
pip install numpy scipy matplotlib qsppack
```

## Reproducing the paper's experiments

```bash
python qsvt_complementary.py
```

This runs all four benchmarks in sequence, prints progress and the residual error for each instance, and saves one figure per benchmark to the working directory:

| Benchmark | Parameters | Output file |
|---|---|---|
| Hamiltonian simulation, cos(τx) | τ = 1000, …, 5000 | `Hamiltonian simulation (Real).pdf` |
| Hamiltonian simulation, sin(τx) | τ = 1000, …, 5000 | `Hamiltonian simulation (Imaginary).pdf` |
| Eigenstate filtering | 1/Δ = 12.5, 25, 50, 100, 200 | `Eigenstate filtering.pdf` |
| Matrix inversion | κ = 16, 64, 256, 1024 | `Matrix Inversion.pdf` |
| Fermi–Dirac operator | β = 100, 200, 400, 800, 1600 | `Fermi-Dirac Operator.pdf` |

Each figure has four panels: (a) the scaled target function, (b) the polynomial degree d, (c) the phase factor construction time, and (d) the phase factor error.

To run a single benchmark, comment out the other calls in the `if __name__ == "__main__":` block at the bottom of the file (`ham_sim()`, `eig_fil()`, `mat_inv()`, `fer_dir()`).

## Using it on your own function

The entry point is `solve`:

```python
info = solve(a, n_samples, parity, thres=1e-12, known_deg=None)
```

| Argument | Meaning |
|---|---|
| `a` | Target function on [-1, 1], vectorized over NumPy arrays. It is rescaled internally so that its maximum absolute value is `NORM_LIMIT` (0.3). |
| `n_samples` | Number of sample points used for the FFT-based polynomial approximation. It should be well above the expected degree. |
| `parity` | `0` for an even function, `1` for an odd one. |
| `thres` | Relative truncation threshold for the Fourier coefficients. |
| `known_deg` | Optional. Fixes the degree instead of choosing it from `thres`; use this when the target is already a polynomial. |

It returns a dictionary:

| Key | Meaning |
|---|---|
| `gammas` | The d + 1 phase factors, as a NumPy array |
| `deg` | Degree d of the polynomial approximation |
| `scale` | Factor by which the input function was rescaled |
| `time` | Phase factor construction time in seconds (excludes the polynomial approximation step) |
| `phase_error` | Estimated relative L∞ error of the reconstructed polynomial |

Example:

```python
import numpy as np
from qsvt_complementary import solve

info = solve(lambda x: np.cos(100 * x), 4000, parity=0)
phases = info["gammas"]
print(info["deg"], info["phase_error"])
```

## How the code is organized

| Stage | Functions | Paper section |
|---|---|---|
| Polynomial approximation by FFT and truncation | `poly_approx` | 3.2 |
| Build 1 − a² and sample h′/h on the unit circle | `comp_sq`, `deriv`, `eval_g` | 3.3 |
| Power sums → characteristic polynomial (power-series exponential by Newton iteration, O(d log d)) | `poly_from_power_sums` | 3.4 |
| Complementary polynomials c and d, normalization α | inside `solve` | 3.5 |
| Divide-and-conquer layer peeling for the phase factors | `phases_from_polys`, `_peel`, `_apply_transfer` | 3.6 |
| Benchmarks and plotting | `plot_graphs`, `ham_sim`, `eig_fil`, `mat_inv`, `fer_dir` | 4 |

Implementation details:

- The logarithmic derivative is sampled at about 40d points on the unit circle (`eta = 0.05` in `solve`).
- Layer peeling switches from the recursive scheme to a direct loop below 64 steps (`_PEEL_BASE`).
- The phase factors are in QSPPACK's full (`typePhi = 'full'`) convention, targeting the real part of the upper-left matrix entry.

## Error measurement

The reported `phase_error` compares the Chebyshev approximation with the polynomial reconstructed from the computed phases by `qsppack.utils.get_entry`. It is evaluated at 20 random points of the sampling grid (fixed seed) plus the maximizer of |a|, and divided by `NORM_LIMIT`. Because only 21 points are used, it is a noisy estimate of the true maximum error.

## Limitations

- The target is always scaled to a maximum of 0.3. The method degrades as the norm approaches 1, because the roots of h approach the unit circle.
- Only real targets of definite parity are supported. Functions of mixed parity must be split into even and odd parts and handled separately.
- Everything runs in double precision. `poly_from_power_sums` raises a `RuntimeError` if the series exponential overflows.

## Citation

If you use this code, please cite:

> Christopher Peng. *QSP Phase Factorization via Logarithmic Derivatives.* Preprint, October 2026.

## Acknowledgements

Thanks to Hongkang Ni for mentorship and feedback. This code builds on the algorithms of Ying (2022) and Ni et al. (2025), and uses QSPPACK for verification.
