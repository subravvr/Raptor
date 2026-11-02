# Ensemble example: stochastic lack-of-fusion at V = 1300 mm/s vs XCT

Each Monte Carlo run simulates a set of 500 µm SVEs (2362 SVEs, 295.25 mm³) and compares defects larger than
20 µm with the two XCT-scanned Ti-6Al-4V 4PB coupons built at V = 1300 mm/s:

* CMU02-05-1, 295.32 mm³, compared with all 2362 SVEs of each run;
* CMU02-32-1, 246.85 mm³, compared with the first 1974 SVEs of each run.

Each comparison uses simulated coupons of the same volume as the scanned one. The quantities compared are:

* **number density (ND)**: defects per mm³ in 2.5 µm equivalent-diameter bins;
* **volume fraction (VF)**: Σ(π/6)d³ / volume of those defects, computed the same way for simulation and
  experiment.

The script reports the ensemble mean with a bootstrap 95% confidence interval (resampling the runs). It also gives
the run-to-run spread and the experiment's percentile within that spread. One run is the simulated counterpart of
one coupon, so the spread is what you would expect from scanning coupons of this size.

## Run

```bash
cd examples/ensemble_example
python run_ensemble.py                         # 20 runs x 2362 SVEs, a few minutes
python run_ensemble.py --n-mc 4 --n-sve 200    # quick look
python run_ensemble.py --plot-only             # re-analyze ensemble_results.npz
```

Outputs: `ensemble_results.npz` (pore diameters for every run), `ensemble_v1300.png`, and a summary table printed
to the console.

## Model settings

| Setting | Value |
|---|---|
| Process | P = 370 W, v = 1.3 m/s, H = 140 µm, L = 30 µm, 67° rotation |
| Effective layer thickness | L / 0.5 = 60 µm (default `powder_consolidation`) |
| Melt-pool width | spectrum `data/meltPoolData/v1300_average_spectrum.csv` (first 50 rows), mean W·λ with W = 172.7 µm, λ = 1.04 |
| Depth / cap | η_D = 0.778, η_C = 0.161 times the width, both parabolic (shape factor 1) |
| Phases | random per scan vector, seed = 13000000 + 10000·run + SVE |
| Porosity | `boundary_tracking=False`: every never-melted voxel is a pore |
| Voxels | 5 µm, 26-connected pores, single voxels removed |

## Data

* `data/meltPoolData/v1300_average_spectrum.csv`: width spectrum averaged over 8 gas-flow angles. Columns are
  amplitude (m) and frequency (Hz); row 0 is the mean. Source: Miner & Narra (2024),
  https://doi.org/10.1184/R1/25696293.v1.
* `data/defectData/v1300_CMU02-05-1_eqd.csv` and `v1300_CMU02-32-1_eqd.csv`: XCT equivalent spherical diameters
  (µm) of the defects in each coupon. Each header records the scanned volume.

## Expected output (20 runs)

```
CMU02-05-1 (295.32 mm^3) vs 20 simulated coupons of 2362 SVEs (295.25 mm^3)
quantity (> 20 um)           run mean  bootstrap 95% CI of mean    per-run 2.5-97.5%            experiment   pctl  in range
ND total (1/mm^3)           3.556e-03  [2.202e-03, 5.080e-03]  [0.000e+00, 1.016e-02]    1.016e-02   90.0  yes
VF (-)                      1.655e-08  [1.008e-08, 2.346e-08]  [0.000e+00, 4.779e-08]    5.856e-08  100.0  no
defects > 20 um per coupon: 1.05 +/- 1.00 (experiment 3)

CMU02-32-1 (246.85 mm^3) vs 20 simulated coupons of 1974 SVEs (246.75 mm^3)
quantity (> 20 um)           run mean  bootstrap 95% CI of mean    per-run 2.5-97.5%            experiment   pctl  in range
ND total (1/mm^3)           3.647e-03  [2.026e-03, 5.274e-03]  [0.000e+00, 1.023e-02]    2.431e-02  100.0  no
VF (-)                      1.697e-08  [9.397e-09, 2.520e-08]  [0.000e+00, 5.068e-08]    1.632e-07  100.0  no
defects > 20 um per coupon: 0.90 +/- 0.97 (experiment 6)
```

* **CMU02-05-1.** The number density is consistent with the ensemble: 3 defects > 20 µm sits at the 90th
  percentile of simulated coupons (1.05 ± 1.00 defects). The volume fraction lies above every run, driven by a
  single 24.2 µm defect; all simulated defects > 20 µm are smaller than 22.5 µm.
* **CMU02-32-1.** Both quantities lie above every simulated coupon of the same volume: 6 defects > 20 µm against
  0.90 ± 0.97, including a 29.6 µm defect. ND is about 7× the ensemble mean and VF about 10×. With these settings,
  the model underpredicts the large-defect tail of this coupon at λ = 1.04.
