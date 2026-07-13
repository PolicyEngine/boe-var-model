# Improvement plan from independent toolbox review

Findings from reviewing bsvarSIGNs, Chan–Matthes–Yu (2025), Arias et al.
(2018/2024), the ECB BEAR toolbox, Cesa-Bianchi's VAR-Toolbox, Cascaldi-Garcia's
pandemic_priors, and GLP (2015) against this replication. Ranked by impact on
replication fidelity.

## Correctness (biases in the current identification step)

1. **Missing importance weights for zero restrictions (highest priority).**
   Arias et al. (2018, Thm. 4) require re-weighting draws from the recursive
   null-space construction by a volume-element correction when zero
   restrictions are present; bsvarSIGNs implements it in
   `src/restrictions_zero.cpp` (`weight_zero`, `log_volume_element`):
   `w = exp(-(2n+m+1)·log|det(A0)| - ½·log det(D_N'D_N))`, with finite-difference
   Jacobians of the zero-restriction map. Our `identify()` treats accepted
   draws as equally likely → biased medians/FEVDs. Fix: return (draw, B, w)
   and use weighted quantiles (or resample ∝ w) in `analysis.py`; report the
   weights' effective sample size.

2. **Per-draw retry bias.** `identify()` retries many Q's per posterior draw
   and keeps the first success; Arias, Rubio-Ramírez, Shin & Waggoner (2024)
   show this targets the wrong distribution. Fix: one Q attempt per posterior
   draw, jointly reject (Σ, Q) on sign failure — affordable once item 3 is in.

3. **Chan–Matthes–Yu (2025) permutation search** (the efficiency device the
   BoE paper says it combines with Arias et al.). The Haar measure is invariant
   under column permutations/sign flips, so given one Q, search all relabelings
   *within blocks sharing identical zero restrictions* — here
   {world_demand, world_energy, unident_global} and
   {uk_demand, uk_supply, uk_monpol, unident_uk} (world_supply fixed) —
   for a sign-satisfying assignment before rejecting (Algorithm 1 +
   Proposition 1; no extra weights needed for the permutation step;
   ~2³·3!·2⁴·4! ≈ 18k equivalent draws per Q).

After 1–3, re-check the UK monetary-policy FEVD share of CPI — items 1 and 2
are the two candidate mechanisms for our discrepancy with the paper (which
reports monetary policy as the largest domestic CPI contributor).

## Prior refinements (bring us in line with GLP / BEAR / pandemic-priors)

4. **Add the dummy-initial-observation (single-unit-root) prior** — the one
   GLP component we lack (GLP turn on both sum-of-coefficients and
   dummy-initial-observation): one dummy row `Y = ȳ'/θ`,
   `X = [ȳ'…ȳ', 1/θ, 0…]`, default θ = 1.

5. **Hyperparameter selection by marginal likelihood.** Fixed λ = 0.2 is the
   GLP mode, but both BEAR (`hogs`) and GLP-style practice optimize
   (λ, μ, θ, φ) via the closed-form NIW marginal likelihood; at minimum report
   the ML-optimal λ.

6. **Expose φ for Covid-dummy columns separately from the constant's
   `eps_const`.** Our plain-dummy treatment is algebraically the Cascaldi-Garcia
   Pandemic Prior at its own uninformative default φ = 0.001 —
   `docs/methodology.md` §7.4 overstates the deviation. A separate φ knob (with
   optional marginal-likelihood grid, à la `OptimalPhi.m`) matches it exactly.

7. **Parameterize the lag-decay exponent** (we hardcode 1/l²; GLP use 2, BEAR
   defaults to 1 with range [1,2]) and optionally δ < 1 own-lag prior mean for
   Bank Rate (BEAR default ar = 0.8). Low priority.

8. **Historical decomposition: report Covid-dummy contributions as an explicit
   separate component** (VAR-Toolbox `compute_HD.m` pattern), like the black
   bars in the paper's Figure 6, instead of folding them into the
   deterministic path.

## Already matching best practice (verified against toolboxes)

- p = 4 for quarterly data (BEAR default), λ = 0.2 (GLP mode), μ = 1
  sum-of-coefficients (GLP default).
- Haar draws via sign-corrected QR (identical to VAR-Toolbox `OrthNorm.m`).
- Zero restrictions via Arias et al. null-space projection (beyond what BEAR
  or VAR-Toolbox offer).
- 68/90 pointwise-median bands (paper/literature convention).
- Covid dummies ≡ Pandemic Prior at φ = 0.001.

## Sources

- https://github.com/bsvars/bsvarSIGNs (esp. `src/restrictions_zero.cpp`,
  `src/sample_Q.cpp`); paper: https://arxiv.org/pdf/2501.16711
- Chan, Matthes & Yu (2025): https://arxiv.org/abs/2503.20668
- https://github.com/european-central-bank/BEAR-toolbox
- https://github.com/ambropo/VAR-Toolbox
- Pandemic Priors: https://www.federalreserve.gov/econres/ifdp/pandemic-priors.htm
  and https://github.com/dcascaldi/pandemic_priors
- GLP (2015): https://www.nber.org/papers/w18467
