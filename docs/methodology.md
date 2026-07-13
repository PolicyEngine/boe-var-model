# Methodology

Writeup of the method in Brignone & Piffer (2025), *A structural VAR model
for the UK economy*, Bank of England Macro Technical Paper No. 3, and of how
this Python replication implements it. Section/equation numbers refer to the
paper (`docs/a-structural-var-model-for-the-uk-economy.pdf`).

## 1. Model

The reduced-form model is a VAR(p) in levels with a constant and Covid-19
dummies (paper eq. 1):

```
y_t = Σ_{l=1..p} Π_l y_{t-l} + c + Σ_{c=1..q} δ_c d_{t-c} + u_t,
u_t ~ N(0, Σ)
```

- `y_t` is k×1 with k = 8 variables (Section 2 below).
- `Π_l` are k×k autoregressive coefficient matrices, `c` a k×1 constant.
- The dummies `d` cover the six quarters 2020Q1–2021Q2 (one dummy per
  quarter, value 1 in that quarter, 0 otherwise), absorbing the extreme
  Covid observations.

Reduced-form innovations map to structural shocks (eqs. 2–3):

```
u_t = B ε_t,      ε_t ~ N(0, I_k),      Σ = B B'.
```

`B` is parameterised through the Cholesky factor of Σ and an orthogonal
matrix Q (eq. 4):

```
B = chol(Σ) · Q,      Q'Q = I.
```

Any orthogonal Q yields the same reduced-form fit; identification consists
of restricting the set of admissible Q.

## 2. Variables and transformations (Table 1)

Quarterly data; estimation sample **1992Q1–2023Q2** (start of UK inflation
targeting; last year of data reserved because recent data are revision-
prone). Data through 2024Q2 are kept for the paper's forecast-revision
exercises. Ordering is fixed — the zero restrictions below reference it.

| # | Variable | Enters model | Shown in figures |
|---|----------|--------------|------------------|
| 1 | Real World GDP (UK-trade-weighted) | 100·log | YoY growth |
| 2 | World CPI (UK-trade-weighted)      | 100·log | YoY growth |
| 3 | Real oil price in sterling         | 100·log | YoY growth |
| 4 | Bank Rate                          | level (%) | level |
| 5 | Sterling exchange-rate index (ERI) | 100·log | level |
| 6 | UK CPISA (seasonally adjusted CPI) | 100·log | YoY growth |
| 7 | UK CPI Energy                      | 100·log | YoY growth |
| 8 | Real UK GDP                        | 100·log | YoY growth |

CPI Energy is included to capture the combined role of oil *and gas* prices
in UK CPI, which helps identify the world energy shock. A fall in the ERI
is a sterling depreciation.

## 3. Identification (Table 2)

Six of the eight shocks are identified: world demand (WD), world energy
(WE), world supply (WS), UK demand (UD), UK supply (US), UK monetary policy
(MP). Two are left unidentified — one global (U1), one UK-specific (U2) —
to absorb residual volatility. Restrictions apply **on impact only**
(no restrictions at longer horizons, and none on B⁻¹).

Reproduction of Table 2 (rows = variables, columns = shocks; blank =
unrestricted):

| Variable    | WD | WE | WS | U1 | UD | US | MP | U2 |
|-------------|----|----|----|----|----|----|----|----|
| World GDP   | +  | +  | +  |    | 0  | 0  | 0  | 0  |
| World CPI   | +  | −  | −  |    | 0  | 0  | 0  | 0  |
| Oil price   |    | −  |    |    | 0  | 0  | 0  | 0  |
| Bank Rate   |    |    |    |    | +  |    | +  |    |
| Exch. rate  |    |    |    |    |    |    |    |    |
| UK CPISA    | +  | −  | −  |    | +  | −  | −  |    |
| CPI Energy  | +  | −  | 0  |    |    |    |    |    |
| Real UK GDP | +  | +  | +  |    | +  | +  | −  |    |

Economic content:

- **Small-open-economy block:** UK shocks (UD, US, MP) and the UK
  unidentified shock (U2) have zero contemporaneous impact on the three
  world variables (rows 1–3). U1, the global unidentified shock, is *not*
  subject to this zero block — that is what makes it "global".
- **World demand (+):** raises world and UK GDP and world and UK CPI.
- **World energy (expansionary/disinflationary):** raises world and UK real
  GDP, lowers world CPI, oil prices, UK CPI and CPI energy.
- **World supply (+):** raises world/UK GDP, lowers world/UK CPI; the zero
  on CPI energy on impact exogenises it with respect to the world energy
  shock.
- **UK demand (+):** raises UK GDP, UK CPI and Bank Rate.
- **UK supply (+):** raises UK GDP, lowers UK CPI.
- **UK monetary policy (tightening):** raises Bank Rate, lowers UK CPI and
  UK GDP.

The paper notes (fn. 3) it does not additionally require that unidentified
shocks avoid repeating the sign patterns of identified ones (impossible for
the UK unidentified shock in this specification), and reports robustness to
dropping the oil price and using a single restricted global unidentified
shock.

## 4. Estimation

Bayesian, following **Giannone, Lenza & Primiceri (2015)**: a Minnesota
(normal-inverse-Wishart) prior on the autoregressive parameters, centred on
a random walk for each variable, combined with the **sum-of-coefficients**
prior (which, per Bergholt et al. 2024, reduces uncertainty around the
deterministic component). Covid-19 is handled with dummies for 2020Q1–2021Q2
in the spirit of the **pandemic prior** of Cascaldi-Garcia (2022). The
authors report robustness to ending the sample before Covid.

In this replication the posterior is the standard conjugate NIW: draw Σ
from an inverse Wishart and coefficients Π from a matrix normal conditional
on Σ (`src/boe_var/bvar.py`).

## 5. Drawing Q — Arias, Rubio-Ramírez & Waggoner (2018)

For each posterior draw (Π, Σ):

1. Compute `L = chol(Σ)`.
2. Draw Q so that the **zero restrictions hold exactly**: build Q column by
   column; for shock j, draw a standard normal vector, project it onto the
   null space of (a) the zero restrictions applying to column j expressed in
   terms of `L·q_j`, and (b) the previously drawn columns; normalise.
3. Compute `B = L·Q` and check the **sign restrictions** column by column.
   Flip a column's sign if that satisfies its restrictions (sign of a
   column of Q is not identified). Accept the draw only if all sign
   restrictions hold; otherwise reject (accept/reject à la Uhlig 2005).

The paper speeds this up by combining Arias et al. (2018) with **Chan,
Matthes & Yu (2025)**: because the uniform (Haar) distribution over
orthogonal matrices is invariant to column permutations and sign flips, all
orderings of the columns of B are searched for one satisfying the sign
restrictions before rejecting, materially reducing the cost of reaching
10,000 accepted draws. (With zero restrictions, permutations are only valid
among columns sharing the same zero pattern — here within {WD, WS, U1... }
blocks; the replication permutes only within restriction-equivalent groups.)

The paper uses **10,000 accepted draws**; this replication targets 1,000+
by default (configurable).

## 6. Outputs and what they mean

- **Impulse responses (Figs. 2–3):** dynamic response of each variable to a
  one-standard-deviation structural shock, computed from the companion form
  of (Π) and impact matrix B; reported as pointwise posterior median with
  68%/90% credible bands across accepted (posterior draw, Q) pairs.
- **Forecast-error variance decompositions (Fig. 4):** share of the
  h-step-ahead forecast-error variance of each variable attributable to
  each shock.
- **Estimated structural shocks (Fig. 5):** `ε_t = B⁻¹ u_t` evaluated at
  each accepted draw; the posterior median series shows when demand/supply/
  policy shocks hit.
- **Historical decompositions (Fig. 6):** additive decomposition of each
  variable's deviation from its deterministic path into cumulative
  contributions of each structural shock (plus initial conditions),
  answering "which shocks explain the data at each date".

## 7. Differences from the paper

Explicit deviations of this replication:

1. **Proxy world series.** The Bank's internal UK-trade-weighted world GDP
   and world CPI are unpublished; we use weighted OECD/IMF aggregates
   (and FRED-sourced components). Global-block results are therefore only
   qualitatively comparable.
2. **Real oil price construction.** Brent USD converted to sterling and
   deflated by UK CPI; the paper does not fully specify its construction.
3. **Lag length assumed p = 4.** The paper does not state p; 4 is standard
   for quarterly data and is a parameter in this code.
4. **Simplified pandemic prior.** We include the six Covid dummies as
   exogenous regressors in the VAR rather than implementing the full
   dummy-observation Minnesota-prior extension of Cascaldi-Garcia (2022);
   this is the same conditioning idea but not identical posterior algebra.
5. **Prior hyperparameters.** GLP (2015) treat prior tightness
   hierarchically; the paper does not report the chosen values. We use
   standard defaults (configurable) rather than hierarchically optimised
   hyperparameters.
6. **Fewer accepted draws.** Target 1,000+ accepted identification draws
   versus the paper's 10,000; credible bands are correspondingly noisier.
7. **Column-permutation search** is implemented only within groups of
   shocks sharing identical zero-restriction patterns (see Section 5),
   which is a conservative reading of the Chan et al. (2025) trick.
8. **Data vintages.** The paper uses the vintages available for the August
   2024 Monetary Policy Report; we use current vintages, so revised data
   will differ, particularly for recent quarters.
9. The forecast-revision tool of Section 5 of the paper (structural
   narrative for round-to-round forecast revisions) is out of scope; we
   replicate the standard outputs (Figures 2–6) only.
