# Handoff notes

Traps and non-obvious facts. Read before changing anything.

## Which code is authoritative

- **`src/paper_models.ipynb` is the current analysis.** Everything in
  `src/multivariate_analysis/`, `src/cnr_analisys/` and `src/summary_analisys/`
  is obsolete, kept for reference. `src/slides_helpers.py` superseded
  `src/multivariate_analysis/_functions.py`.
- `archive/` holds files that are dead — see `archive/README.md` for why each one.

## Working directory

Notebooks and most scripts use relative paths and CWD-based imports. **Run each
from the directory it lives in.** Three different `sys.path` hacks are in play:

| File | Hack |
|---|---|
| `src/reliability/Reliability.ipynb` | `sys.path.insert(0, os.path.abspath("../.."))` then `from src.functions… import` |
| `src/cnr_analisys/cnr_analysis.ipynb` | `sys.path.insert(0, '../summary_analisys')` |
| `src/surface_analysis/thickness_analysis.ipynb` | `sys.path.insert(0, '..')` for `slides_helpers` |
| `src/paper_models*.ipynb`, `bootstrap_covariates.ipynb` | no hack — rely on CWD being `src/` |

Consequences: the directory names `cnr_analisys/` and `summary_analisys/` **cannot be renamed** without breaking the second hack. And
`src/functions/CNR.py` uses a relative import, so it must be run as
`uv run python -m src.functions.CNR` from the repo root.

## CSVs that get rewritten in place

These read a CSV, add columns and write it back. Run them in pipeline order or
you will lose columns:

| Writer | Rewrites |
|---|---|
| `src/functions/CNR.py` | `data/image_quality_metrics.csv`, `data/split_comparision_data.csv` |
| `src/surface_analysis/surface_analisys.ipynb` | `data/image_quality_metrics.csv` (adds `wm_surface_area`) |
| `src/reliability/Reliability.ipynb` | `data/cross_split_metrics.csv`, `data/image_quality_metrics.csv` |
| `src/bootstrap_covariates.ipynb` | `assets/reliability_summary_table.csv` — **must run after `paper_models.ipynb`**, which creates it |

## Data you cannot regenerate

`data/subject.csv` and `data/raw_subject_data.csv` are obtained obtained from the reliability project test cohort. [This spreadsheet](https://bostonchildrenshospital-my.sharepoint.com/:x:/r/personal/yair_beltran_childrens_harvard_edu/Documents/Attachments/Copy%20of%20reliability%20test_SJ.xlsx?d=wf840aa38f7b94b429c20465c5581eb14&csf=1&web=1&e=NR61XF)
 contains most fo the information on both files. If you dont have access, request the original spreadsheet from [Andrea Gondova](mailto:Andrea.Gondova@childrens.harvard.edu).

`wm_surface_area_diff` / `wm_surface_area_mean` in `split_comparision_data.csv`
are produced by calculating the gaussian distance between the vertex points of the WM surface area and the inner SP surface area. Both files can be found on each subject, more [bellow](#subject-directory-layout).

## Lab network access

- Stages 1–3 need `/neuro/...` access; See `docs/environment.md` for the full path inventory.
- `run_SP_prediction.py` needs its own conda env and does not run on el-jobo. Consult [Andrea Gondova](mailto:Andrea.Gondova@childrens.harvard.edu).

## Subject directory layout
