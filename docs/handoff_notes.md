# Handoff notes

Traps and non-obvious facts. Read before changing anything.

## Which code is authoritative

- **`src/paper_models.ipynb` is the current analysis.** Everything in
  `src/multivariate_analysis/`, `src/cnr_analisys/` and `src/summary_analisys/`
  is an earlier generation, kept for reference. `src/slides_helpers.py` superseded
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

Consequences: the directory names `cnr_analisys/` and `summary_analisys/` are
misspelled but **cannot be renamed** without breaking the second hack. And
`src/functions/CNR.py` uses a relative import, so it must be run as
`uv run python -m src.functions.CNR` from the repo root — its data paths are
`__file__`-anchored to make that work from anywhere.

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

`data/subject.csv` and `data/raw_subject_data.csv` are hand-entered (the QA
scores are manual image assessments). They are tracked in git for exactly this
reason. Do not add them to `.gitignore`.

`wm_surface_area_diff` / `wm_surface_area_mean` in `split_comparision_data.csv`
have no producing script — the derivation is documented and verified in
`docs/data_dictionary.md`, but nothing regenerates them automatically.

## Cluster coupling

- Stages 1–3 need `/neuro/...` access; stage 4 (all the paper notebooks) runs
  from a clone. See `docs/environment.md` for the full path inventory.
- `src/processing/extract_SP_surface.py` is **not** the file that runs. Its caller
  `run_SP_batch.py` invokes a copy at
  `/neuro/labs/grantlab/research/MRI_processing/yair.beltran/Reliability/extract_SP_surface.py`.
  Edits must be copied over.
- `run_SP_prediction.py` needs its own conda env and does not run on el-jobo.

## Known rough edges

- `src/image_quality_metrics.py` writes long-format CSVs but `add_derived_metrics()`
  also builds a wide format that nothing currently consumes.
- `src/surface_analysis/thickness_analysis.ipynb` has its cell set **duplicated** —
  the whole analysis appears twice. Any edit must be made in both copies.
- `src/surface_analysis/surface_analisys_summary.ipynb` re-defines a local copy of
  `_compute_split_pair_diff` instead of importing it from
  `src/summary_analisys/_functions.py`.
- Repo-wide `ruff check` reports ~80 pre-existing style errors, almost all in
  notebooks. The processing and metrics scripts are clean.
