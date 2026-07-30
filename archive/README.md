# archive/

Unmaintained. Nothing here is part of the pipeline — kept only so old numbers can be traced.

| File | Why it is here |
|---|---|
| `Reliability.py` | `nbconvert` dump of `src/reliability/Reliability.ipynb`. The notebook is authoritative. |
| `multivariate_analysis.py` | `nbconvert` dump; imports `split_comp_df` from `_functions`, which no longer defines it. Broken. |
| `multivariate_analysis.nbconvert.ipynb` | Execution artifact of `src/multivariate_analysis/multivariate_analysis.ipynb`. |
| `multivariate_analisys_old.ipynb` | Pre-`src/` version, reads `../data/…` paths that no longer resolve. |
| `native_volume.ipynb` | Empty stub (one markdown title, one blank cell). |
| `cnr_analysis.nbconvert.ipynb` | Execution artifact of `src/cnr_analisys/cnr_analysis.ipynb`. |
| `_temp_exec.ipynb` | Byte-identical scratch copy of `src/cnr_analisys/cnr_multivariate_analysis.ipynb`. |
| `cnr_multivariate_plots.py` | Reads `assets/cnr_regression_results.pkl`, which nothing in the repo writes. Unrunnable. |
| `Untitled.ipynb` | Empty. |
| `reliability_paper_textdump.txt` | Plain-text extraction of `assets/reliability_paper.pdf`. Use the PDF. |

`data/archive/` holds the matching stale data: superseded IQM backups, `cross_split_metrics_old.csv`, and `failed.csv` (a scratch log from `src/processing/run_SP_batch.py`).
