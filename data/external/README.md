# External records (append-only)

Downloaded 19 Nov 2026 from official publishers only (Vahid approved: the November 2026 MPR comparator, the data to score the 2026Q3 forecasts, and source-level seasonally adjusted euro-area and German CPI). Each raw file was saved to its own new directory outside the repo and read as untrusted data with `python -I`. Only the values used are committed here; the raw files are not, and `data/boe_var_data.csv` is not changed. Append new files or rows; do not edit these.

| File | Contents | Source (URL) | Release | Raw file SHA-256 |
|---|---|---|---|---|
| `boe_mpr_2026_11_table_3b.csv` | November 2026 MPR Table 3.B, central projection, with the July 2026 equivalents shown in parentheses in the table. Hand-transcribed and checked against a rendering of page 75. | https://www.bankofengland.co.uk/-/media/boe/files/monetary-policy-report/2026/november/monetary-policy-report-november-2026.pdf | 5 Nov 2026 | `c0f5c8de9faac850a869c3de3bbcf674f6926ed9b25e43a10dc450b381ea8018` |
| `uk_outturns_2026q3.csv` | GDP ABMI (PN2, the first quarterly estimate) | https://www.ons.gov.uk/generator?format=csv&uri=/economy/grossdomesticproductgdp/timeseries/abmi/pn2 | 12 Nov 2026 | `119d30304b844b26114b8b17de6b9aa33e2db2f53c67030637ee3c84f3d3d835` |
| | CPI D7BT, monthly (MM23) | https://www.ons.gov.uk/generator?format=csv&uri=/economy/inflationandpriceindices/timeseries/d7bt/mm23 | 18 Nov 2026 | `fe1d758dfed5205f458062a27097983e266989c8a2d262642b80c4e827a3bfab` |
| | Unemployment MGSC, MGSX (LMS) | …/timeseries/mgsc/lms, …/timeseries/mgsx/lms (`employmentandlabourmarket/peoplenotinwork/unemployment`) | 17 Nov 2026 | MGSX `d058f9cc72b74205ecce678e8e4fda0db6dfbcc7dcffba86681e672a8c71332a` |
| | Employment MGRZ (LMS) | …/timeseries/mgrz/lms (`employmentandlabourmarket/peopleinwork/employmentandemployeetypes`) | 17 Nov 2026 | see notes |
| | Bank Rate IUQABEDR, quarterly average | https://www.bankofengland.co.uk/boeapps/iadb/fromshowcolumns.asp (same query as `data/README.md`, from 2024) | retrieved 19 Nov 2026 | `7843504881a6e1b16629ee8d8bb6e84d9f5261dbf9e39b74dd8f72473b93617a` |
| `ecb_hicp_euro_area_monthly.csv` | Euro area (changing composition) HICP, overall index, 2025 = 100: working-day and seasonally adjusted (ECB, `HICP.M.U2.Y.000000.4F0.INX`) and unadjusted (Eurostat via the ECB portal, `HICP.M.U2.N.000000.4D0.INX`), 1997-01 to 2026-10 | https://data-api.ecb.europa.eu/service/data/HICP/M.U2.Y.000000.4F0.INX?format=csvdata and …/M.U2.N.000000.4D0.INX?format=csvdata | retrieved 19 Nov 2026 | SA `6115aa0a2d0947c8a2f9c25d6e2682b41674da39871ad25445b9dd4ab4b46816`, NSA `18cd12c694753b517fb80e02ff986a9b6a32d81fe62d994e8838a6275f09b580` |
| `svar_stored_forecasts_2026q3.csv` | The 2026Q3 entries of the two archived site rounds that forecast that quarter, copied unchanged (source path and commit in each row) | PolicyEngine/macro `forecasts/rounds/` | archived 29 Jul and 12 Nov 2026 | n/a |

Notes
- The ECB's older adjusted key `ICP.M.U2.Y.000000.3.INX` stops at 2025-12 (superseded by the 2025 = 100 `HICP` dataset); it was downloaded and not used.
- **German CPI seasonally adjusted at source: not obtained.** Destatis serves its tables through GENESIS, whose download service needs an account login; it was not used. The ECB portal has no adjusted HICP for Germany (`HICP.M.DE.Y.000000.4F0.INX` returns 404). The German series only backcasts the euro-area leg before 1997, so the check in `results/ecb_seasonal_check.md` starts in 1997Q2.
- MGSC and MGRZ hashes: MGSC `2d1417ac9a45cea5ef622ded4c16656e2c730579c1227e212b30b7d8cd53349d`, MGRZ `c194fecaebac15c9501520447738b923b7290e8dac83c79364ee770bd5275033`.
- Used by `scripts/score_2026q3.py` (→ `results/score_2026q3.md`), `scripts/compare_ecb_seasonal.py` (→ `results/ecb_seasonal_check.md`) and the November MPR section of `results/forecast_vs_boe_obr.md`.
