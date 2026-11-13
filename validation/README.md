# Validation of the Python SUR criterion against KrigInv

Setting: R 4.3 (conda env `rkrig`), KrigInv 1.4.2 (GitHub mirror cran/KrigInv), DiceKriging 1.6.0.

* `kriginv_reference.R` : fixed-hyperparameter DiceKriging model (Matern 5/2, d=2, n=20), KrigInv's
  `sur_optim_parallel` evaluated on 40 candidates / 400 integration points; CSVs in `kriginv_ref/`.
* `validate_sur_vs_kriginv.py` : max relative difference closed form vs KrigInv 1.1e-15;
  Gauss-Hermite(60) 4.6e-4; same argmin; posterior mean/sd/covariances recomputed in numpy agree to ~1e-12.
* `sur_restrictions_study.py` / `sur_restrictions.csv` : 18 designs (branch k=6 and oscillator, n in
  {30,50,80}, 3 seeds, pool 5000). Candidate subset (200, 1000), active-point truncation (|m|<=6s) and the
  closed form give the same selected point as the full reference in 18/18 designs; 15-node Gauss-Hermite
  differs in 6/18 (max regret 1.5% of the reference gain), 40 nodes in 0/18.

Known difference: KrigInv uses universal kriging (trend-estimation uncertainty); the Python GP treats
the fitted constant mean as known.
