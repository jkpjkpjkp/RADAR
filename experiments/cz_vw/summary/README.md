# Completed bounded protocol result

The one-window, seed-42 run produced **positive estimated alpha, but no
statistically significant evidence of alpha**. This is an adaptation of RADAR,
not an exact reproduction of the paper's unreleased 377-stock experiment.

| Portfolio | Annual alpha | HAC t | Two-sided p | 95% CI | Holm-adjusted p |
|---|---:|---:|---:|---:|---:|
| Gross | 11.72% | 1.36 | 0.174 | −5.20% to 28.63% | 0.347 |
| After 10 bp per traded dollar | 7.41% | 0.86 | 0.392 | −9.61% to 24.44% | 0.392 |

These are annual arithmetic FF5+UMD intercepts with Bartlett HAC(20), finite-sample
correction and t-reference confidence intervals. The sample is 251 realized return
days, 2015-06-08 through 2016-06-03. Cap-weighted D10−D1 targets +100% long and −100%
short, or 200% gross exposure. Do not interpret this single year and seed as robust
profitability evidence.

Training used the released gross-return objective and stopped after 14 epochs,
with best validation loss 0.0007840488. The model had 202 CZ characteristics and
FinBERT news inputs. Feature calibration predates the experiment; prior-month-end
NYSE20 exclusion, dated PERMNO/news mapping, execution delay, and the other protocol
assumptions are documented in [the experiment guide](../README.md).

Gross cumulative return was 18.04%; after-cost cumulative return was 13.06%.
There were no missing held returns or unliquidated terminal holdings in this run.
Borrow fees, financing, impact and collateral interest are not modeled.

## Provenance and evaluation repair

The run used training code at commit `51849a58f0d66f026931fecd630276ad0effd5fb`.
Evaluation initially encountered two serialization-only problems: NumPy string
dates in the checkpoint and a tuple/list mismatch when comparing the JSON run
lock. The saved checkpoint's date metadata was converted to Python strings, and
evaluation resumed after verifying the original code/input hashes. All 41 model
tensors were unchanged; training was not repeated. The subsequent local code fix
prevents both issues in new runs and includes a regression test that resumes
without retraining or altering the checkpoint.

`alpha.csv` preserves unrounded estimates and `performance.json` contains aggregate
accounting diagnostics. `checkpoint_conversion.json` records both checkpoint
hashes. Raw CRSP/CZ/news data, security-level predictions, and model weights are
not included in this repository.
