# Data dictionary

All CSVs are **long format**: one row per subject×split, or per subject×split-pair.
Join keys are always `subject_id` + `session_id` (subjects can have several sessions).

Splits are `S1 S2 S3 S4`; the two comparison pairs are `S1-S2` and `S3-S4`.
Tissue labels: subplate `4, 5`; cortical plate `1, 42`; inner zone `160, 161`.

---

## `data/subject.csv` — 21 rows

Hand-curated. Master roster; the `--subjects` argument for every processing script.

`subject_id`, `session_id`, `GA` (gestational age, weeks)

## `data/raw_subject_data.csv` — 21 rows

Hand-curated. QA scores are manual image assessments; there is no script for them.

`subject_id`, `session_id`, `stack_count` (imaging stacks, 3–11), `stacks_paired`,
`GA`, `QA_S1`…`QA_S4`, `QA12_mean` (mean QA over S1,S2), `QA34_mean`

## `data/image_quality_metrics.csv` — 91 rows (subject × split)

Produced by **`src/image_quality_metrics.py`**, then enriched by two other steps.

| Columns | Producer |
|---|---|
| `subject_id`, `session_id`, `GA`, `split` | `src/image_quality_metrics.py` |
| `snr_subplate`, `snr_cortical_plate`, `snr_inner` | `compute_snr()` — mean/σ within the tissue mask |
| `cnr_sp_cp`, `cnr_sp_inner`, `cnr_cp_inner` | `compute_cnr()` |
| `sp_*`, `cp_*`, `inner_*` (`mean_intensity`, `std_intensity`, `volume_voxels`) | `compute_tissue_stats()` |
| `scaling_factor` | `compute_scaling_factor()` — from `recon_segmentation/recon_to31_inv.xfm` |
| `native_vol_sp`, `native_vol_cp`, `native_vol_inner` | `compute_native_volume()` = voxel count × scaling factor |
| `sp_iz_cnr` | **`src/functions/CNR.py`** (`uv run python -m src.functions.CNR`) — boundary contrast ratio between the SP and IZ bands |
| `wm_surface_area` | **`src/surface_analysis/surface_analisys.ipynb`** — merged from `surface_audit.total_wm_area`. The notebook rewrites this CSV in place. |

## `data/cross_split_metrics.csv` — 413 rows (subject × model × split pair)

Produced by **`src/reliability/Reliability.ipynb`** (batched via `batch_process.py`).

`subject_id`, `session_id`, `model` (segmentation model compared), `labels` (tissue
label set), `split1`, `split2`, `dice`, `jaccard`, `relative_diff`

## `data/surface_audit.csv` — 362 rows (subject × split)

Produced by **`src/processing/surface_audit.py --batch`**. Requires cluster access.

| Columns | Meaning |
|---|---|
| `split_dir_exists`, `cp_dir_exists`, `sp_dir_exists` | which output dirs the pipeline produced (`surfaces/` = CP, `default_surfaces/` = SP) |
| `cp_status`, `cp_critical_missing`, `cp_measurements_missing`, `cp_vertex_maps_missing`, `cp_templates_missing`, `cp_n_files_total` | CP pipeline completeness |
| `sp_status`, `sp_critical_missing`, `sp_optional_missing`, `sp_has_log`, `sp_log_status`, `sp_n_files_total`, `sp_error_log`, `sp_error_snippet` | SP pipeline completeness + failure reason |
| `lh_sp_area_sum`, `rh_sp_area_sum`, `total_sp_area` | SP surface area, summed triangle areas of `*.innersp.gii` |
| `lh_area_sum`, `rh_area_sum`, `lh_depth_mean`, `rh_depth_mean`, `lh_curv_absmean`, `rh_curv_absmean`, `lh_thk_mean`, `rh_thk_mean` | parsed from `Area_Depth_aMC_Thk.txt` |
| `gi_raw` | parsed from `GI_info_final.txt` |
| `*_wm_{obj,asc,gii}_{size_mb,path}` | WM surface file inventory |
| `GA` | joined from `subject.csv` |

Note: `total_wm_area` referenced by `surface_analisys.ipynb` is derived in-notebook
from `lh_area_sum + rh_area_sum`.

## `data/thickness_audit.csv` — 80 rows (subject × split)

Produced by **`src/processing/thickness_audit.py`**. Subjects filtered to `GA <= 32`.

| Columns | Producer |
|---|---|
| `xfm_scale_x/y/z` | `read_native_xfm()` — diagonal of `recon_segmentation/recon_native.xfm` |
| `{lh,rh}_thk_{n_vertices,mean,median,std,p95,max}` | per-vertex Euclidean distance between `{hemi}.wm..obj` and `{hemi}.innersp.obj` (vertex-corresponded meshes) |
| `wholebrain_thk_mean` | vertex-weighted mean over both hemispheres |
| `GA` | joined from `subject.csv` |
| `split_pair`, `thk_diff`, `thk_mean`, `apd_thk`, `qa_mean`, `stack_count` | `add_pair_columns()` in the same script — see below |

Pair columns, written onto **both** split rows of a pair (the shape
`paper_models*.ipynb` expects), from `a`, `b` = the pair's two `wholebrain_thk_mean`:

```
thk_diff = |a - b|
thk_mean = (a + b) / 2
apd_thk  = thk_diff / thk_mean * 100        # absolute percent difference
qa_mean  = QA12_mean (S1-S2) or QA34_mean (S3-S4), from raw_subject_data.csv
stack_count                                  from raw_subject_data.csv
```

These six were originally added by hand outside any script. `add_pair_columns()`
was written to reproduce the committed values and does so to float precision
(max deviation 2e-14).

## `data/split_comparision_data.csv` — 46 rows (subject × split pair)

The main modelling table. Assembled by three separate producers.

| Columns | Producer |
|---|---|
| `subject_id`, `session_id`, `GA`, `split_pair`, `abs_diff_{sp,cp,inner,total}`, `rel_diff_{sp,cp,inner,total}` | `compute_independent_pair_diffs()` in `src/image_quality_metrics.py`, wired up behind `--pair_diffs`. Absolute and percent differences of `native_vol_*` between the pair's two splits; `total` = sp + cp + inner. |
| `cnr_diff`, `cnr_mean` | `src/functions/CNR.py` `__main__` — abs difference and mean of `sp_iz_cnr` across the pair |
| `wm_surface_area_diff`, `wm_surface_area_mean` | **No script.** Added ad hoc. Verified derivation from `image_quality_metrics.wm_surface_area`: `diff = |a - b|`, `mean = (a + b) / 2` over the pair's two splits (reproduces the committed values to 4e-12). |

> `--pair_diffs` drops and re-merges only the columns it owns, so rerunning it
> preserves `cnr_*` and `wm_surface_area_*`. Verified: the regenerated volume
> columns match the committed file to 3e-13.

---

## Result tables — `assets/`

Written by the stage-4 notebooks; these are the numbers reported in the paper.
See `docs/pipeline.md` for which notebook writes which file.

`reliability_summary_table*.csv`, `model_comparison*_table*.csv`,
`outliers_per_model.csv`, `common_outliers.csv`, `bootstrap_*.csv`, and the
per-model/per-pair regression tables `{vol_M,surf_S,thk_T}_{GA,acq,SNR,CNR,SH}_{S1S2,S3S4}_table.csv`
(emitted by `fit_and_present()` in `src/slides_helpers.py`).
