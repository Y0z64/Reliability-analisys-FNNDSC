# Pipeline

Stages run in order. Stages 1–3 need FNNDSC cluster access; stage 4 runs from a clone.

**Working directory matters.** Notebooks and most scripts use relative paths, so
run each one *from the directory it lives in*. Commands below show the `cd`.

Cluster data root (`BASE_PATH` in every processing script):
`/neuro/labs/grantlab/research/MRI_processing/seungyoon.jeong/2025/Reliability/TEST/`
laid out as `<subject_id>/<session_id>/<split>/…` with splits `S1 S2 S3 S4`.

---

## Stage 0 — hand-curated inputs

No script produces these. If they are lost they must be re-entered by hand.

| File | Contents |
|---|---|
| `data/subject.csv` | Master roster: `subject_id`, `session_id`, `GA`. Every script takes this as `--subjects`. |
| `data/raw_subject_data.csv` | `stack_count`, `stacks_paired`, `GA`, `QA_S1..S4`, `QA12_mean`, `QA34_mean`. QA scores are manual image assessments. |

## Stage 1 — cluster processing

Writes segmentations and surfaces onto the cluster filesystem, not into this repo.

```bash
cd src/processing
uv run python run_SP_prediction.py --subjects ../../data/subject.csv   # SP segmentations
uv run python run_SP_batch.py      --subjects ../../data/subject.csv   # SP surfaces (fans out extract_SP_surface.py)
uv run python generate_native_surfaces.py                              # native-space *.native.obj / *.innersp.native.thk
```

`run_SP_prediction.py` needs a different conda env (see `docs/environment.md`) and
does not run on el-jobo. Both batch scripts write a `failed.csv` into the working
directory. `extract_SP_surface.py` is invoked by `run_SP_batch.py`, not directly —
note it calls a **copy of itself** at
`/neuro/labs/grantlab/research/MRI_processing/yair.beltran/Reliability/extract_SP_surface.py`,
so edits here must be copied there to take effect.

## Stage 2 — audits and metrics → `data/`

```bash
cd src
uv run python image_quality_metrics.py \
    --subjects ../data/subject.csv \
    --base_path /neuro/labs/grantlab/research/MRI_processing/seungyoon.jeong/2025/Reliability/TEST/ \
    --pair_diffs                                     # → data/image_quality_metrics.csv
                                                     # → data/split_comparision_data.csv (volume diff columns)

cd processing
uv run python surface_audit.py --batch               # → data/surface_audit.csv
uv run python thickness_audit.py                     # → data/thickness_audit.csv

cd ../..
uv run python -m src.functions.CNR                   # → sp_iz_cnr in image_quality_metrics.csv
                                                     # → cnr_diff, cnr_mean in split_comparision_data.csv
                                                     # → assets/artifacts/cnr_batch_tests/*.png
```

`src/functions/CNR.py` is the one script that must run as a module from the repo
root — it uses a relative import and resolves data paths from `__file__`.

## Stage 3 — per-subject reliability reports

```bash
cd src/reliability
uv run python batch_process.py     # papermill-drives Reliability.ipynb once per subject
```

Reads `data/subject.csv`; writes `data/cross_split_metrics.csv` (Dice, Jaccard,
relative volume difference), appends to `data/image_quality_metrics.csv`, and puts
one executed notebook + one 2-page PDF per subject in `reports/` (gitignored).

To run a single subject interactively, open `Reliability.ipynb` and edit the
parameters in cell 2.

## Stage 4 — analysis notebooks

Read `data/*.csv` only. **No cluster access needed** — these run from a fresh clone.

Run in this order:

```bash
cd src
uv run jupyter lab      # then run the notebooks, or:
uv run jupyter nbconvert --to notebook --execute --inplace paper_models.ipynb
```

| # | Notebook | Reads | Writes |
|---|---|---|---|
| 1 | `src/paper_models.ipynb` | `raw_subject_data`, `split_comparision_data`, `image_quality_metrics`, `thickness_audit` | `assets/reliability_summary_table.csv`, `model_comparison_table.csv`, `outliers_per_model.csv`, `common_outliers.csv`, the six `vol_M_*`/`surf_S_*`/`thk_T_*` pair tables, `assets/artifacts/model_comparison_R2.png` |
| 2 | `src/paper_models_thickness_subjects.ipynb` | same | `assets/reliability_summary_table_thk_subjects.csv`, `model_comparison_table_thk_subjects.csv`, `assets/artifacts/model_comparison_R2_thk_subjects.png` |
| 3 | `src/paper_models_acq_qa_variants.ipynb` | same | `assets/model_comparison_{noqa,qadiff}_table.csv` + matching PNGs in `assets/artifacts/` |
| 4 | `src/bootstrap_covariates.ipynb` | same + `assets/reliability_summary_table.csv` | `assets/bootstrap_paper4_summary.csv`, `bootstrap_sp_covariate_screen.csv`, **rewrites** `assets/reliability_summary_table.csv` |

> **Ordering constraint:** `paper_models.ipynb` must run before
> `bootstrap_covariates.ipynb`. The latter reads `reliability_summary_table.csv`,
> merges bootstrap columns into it and writes it back. Running bootstrap first fails.

Supporting notebooks (any order, after stage 2):

| Notebook | Purpose |
|---|---|
| `src/surface_analysis/thickness_analysis.ipynb` | Whole-brain SP thickness + thickness reliability models. Run from `src/surface_analysis/`. |
| `src/surface_analysis/surface_analisys.ipynb` | Surface pipeline failure breakdown from `surface_audit.csv`. **Also rewrites `data/image_quality_metrics.csv`** to merge in `wm_surface_area`. |
| `src/surface_analysis/surface_analisys_summary.ipynb` | WM surface area vs GA plots. |
| `src/regional_thickness_demo.ipynb` | Maps per-vertex SP thickness onto the fetal21 atlas for one example subject. Needs cluster access. |

Legacy, superseded by `paper_models.ipynb` — kept for reference only:
`src/multivariate_analysis/`, `src/cnr_analisys/`, `src/summary_analisys/`.

---

## Where things land

| Path | Contents |
|---|---|
| `data/` | Input and derived CSVs. Tracked in git. |
| `assets/` | Result tables (the numbers behind the paper) + the reference-paper PDFs. |
| `assets/artifacts/` | Generated figures. **Gitignored** — every one is regenerated by a notebook, so a fresh clone starts empty here. |
| `reports/` | Per-subject PDFs and executed notebooks. Gitignored. |
| `reference/` | `FreeSurferColorLUT.txt`. |
| `archive/`, `data/archive/` | Dead code and superseded data. See `archive/README.md`. |
