# Reliability Analysis Quick Reference

Fetal brain subplate (SP) segmentation reliability, FNNDSC. One scan per subject
is reconstructed four times (S1, S2, S3, S4) and the outputs are compared.

Contacts: Yair Beltran (yairprogrammer@gmail.com), Andrea Gondova (Andrea.Gondova@childrens.harvard.edu), Seungyoon Jeong.

## Conventions

- Splits: `S1 S2 S3 S4`. Only the non overlapping pairs `S1-S2` and `S3-S4` are modelled.
- Tissue labels: subplate `4, 5`; cortical plate `1, 42`; inner zone `160, 161`.
- All CSVs are long format. Join on `subject_id` + `session_id`.
- Sample: 20 subjects, GA 22 to 32 weeks. Thickness work uses a
  13 subject subset per pair.

## Where the data lives

Cluster data root:

```
/neuro/labs/grantlab/research/MRI_processing/seungyoon.jeong/2025/Reliability/TEST/
```

Per subject layout example:

```
<subject_id>/<session_id>/<split>/
  recon_segmentation/
    recon_to31_nuc.nii                                T2 image
    <subject>_<session>_nuc_deep_subplate_dilate_mc.nii   segmentation
    recon_to31_inv.xfm                                recon to native, used for scaling
    recon_native.xfm                                  native transform
  surfaces/              cortical plate surfaces
  default_surfaces/      subplate surfaces
    {lh,rh}.wm..obj              outer SP boundary (WM surface)
    {lh,rh}.innersp.obj          inner SP boundary
    {lh,rh}.*.native.obj         native space versions
    {lh,rh}.innersp.native.thk   per vertex SP thickness
    Area_Depth_aMC_Thk.txt, GI_info_final.txt
```

Other lab paths:

| Path | Used for |
|---|---|
| `.../milton.candela/fetal_subplate/models/C120` | subplate model weights |
| `.../yair.beltran/Reliability/FreeSurferColorLUT.txt` | label lookup (local copy in `reference/`) |
| `.../yair.beltran/Reliability/extract_SP_surface.py` | copy of the repo script, edits must be mirrored |
| `/neuro/users/mri.team/fetal_mri/Surface_template/template-29/old/lh.fetal21.1D.dset` | atlas for regional thickness |

`...` is `/neuro/labs/grantlab/research/MRI_processing`.

## CSV files in `data/`

| File | Grain | Contents |
|---|---|---|
| `subject.csv` | subject | `subject_id`, `session_id`, `GA`. Passed as `--subjects` to every script. |
| `raw_subject_data.csv` | subject | `stack_count`, `stacks_paired`, `GA`, `QA_S1..S4`, `QA12_mean`, `QA34_mean` |
| `image_quality_metrics.csv` | subject x split | SNR, CNR, tissue stats, volumes, `sp_iz_cnr`, `wm_surface_area` |
| `cross_split_metrics.csv` | subject x model x pair | `dice`, `jaccard`, `relative_diff` |
| `surface_audit.csv` | subject x split | surface pipeline completeness, areas, depth, curvature, GI |
| `thickness_audit.csv` | subject x split | per hemisphere thickness stats plus pair columns |
| `split_comparision_data.csv` | subject x pair | volume, CNR and surface area differences. Main modelling table. |

`subject.csv` and `raw_subject_data.csv` come from the cohort spreadsheet and
cannot be regenerated. Request access from Andrea Gondova
(Andrea.Gondova@childrens.harvard.edu).

## Metric definitions

| Metric | Definition | Written by |
|---|---|---|
| `snr_<tissue>` | mean intensity / standard deviation inside the tissue mask | `src/image_quality_metrics.py` |
| `cnr_a_b` | \|mean_a - mean_b\| / pooled standard deviation | `src/image_quality_metrics.py` |
| `sp_iz_cnr` | contrast ratio, (mean_SP - mean_IZ) / (mean_SP + mean_IZ), measured only on the dilated SP and IZ inner bands | `src/functions/CNR.py` |
| `scaling_factor` | absolute determinant of the 3x3 part of `recon_to31_inv.xfm` | `src/image_quality_metrics.py` |
| `native_vol_<tissue>` | voxel count x voxel volume x scaling factor | `src/image_quality_metrics.py` |
| `dice` | 2 x intersection / (sum of both masks) | `src/functions/helpers.py` |
| `jaccard` | intersection / union | `src/functions/helpers.py` |
| `relative_diff` | \|count1 - count2\| / mean count x 100 | `src/functions/helpers.py` |
| `hausdorff` | max of the two directed Hausdorff distances | `src/functions/helpers.py` |
| `{lh,rh}_thk_*` | per vertex Euclidean distance between `wm..obj` and `innersp.obj` | `src/processing/thickness_audit.py` |
| `wholebrain_thk_mean` | vertex weighted mean of both hemispheres | `src/processing/thickness_audit.py` |
| `total_sp_area` | sum of triangle areas of `*.innersp.gii`, both hemispheres | `src/processing/surface_audit.py` |
| `wm_surface_area` | `lh_area_sum + rh_area_sum` from the surface audit | `src/surface_analysis/surface_analisys.ipynb` |

Pair level metrics, with `a` and `b` the two splits of a pair:

| Metric | Definition |
|---|---|
| `abs_diff_<tissue>` | \|a - b\| of `native_vol_<tissue>`. `total` is sp + cp + inner. |
| `rel_diff_<tissue>` | percent difference of `native_vol_<tissue>` |
| `thk_diff`, `thk_mean` | \|a - b\| and (a + b) / 2 of `wholebrain_thk_mean` |
| `apd_thk` | `thk_diff / thk_mean x 100` |
| `cnr_diff`, `cnr_mean` | \|a - b\| and mean of `sp_iz_cnr` |
| `wm_surface_area_diff`, `wm_surface_area_mean` | \|a - b\| and mean of `wm_surface_area` |
| `qa_mean` | `QA12_mean` for S1-S2, `QA34_mean` for S3-S4 |

Covariates used in the models: GA, `qa_mean`, `stack_count`, SNR, CNR.

## Running the analysis

The notebooks read only `data/*.csv`, which is tracked, so no cluster access is
needed for this part.

```bash
uv sync
cd src
uv run jupyter lab
```

Run `paper_models.ipynb` first, then `bootstrap_covariates.ipynb`. The second one
reads and rewrites `assets/reliability_summary_table.csv`, so running it first
fails.

| Notebook | Writes to `assets/` |
|---|---|
| `paper_models.ipynb` | `reliability_summary_table.csv`, `model_comparison_table.csv`, `outliers_per_model.csv`, `common_outliers.csv`, the `vol_M_*` / `surf_S_*` / `thk_T_*` pair tables |
| `paper_models_thickness_subjects.ipynb` | `*_thk_subjects.csv` variants |
| `paper_models_acq_qa_variants.ipynb` | `model_comparison_{noqa,qadiff}_table.csv` |
| `bootstrap_covariates.ipynb` | `bootstrap_paper4_summary.csv`, `bootstrap_sp_covariate_screen.csv` |

Figures go to `assets/artifacts/`, which is gitignored and regenerated.

Supporting notebooks, any order:

| Notebook | Purpose |
|---|---|
| `src/surface_analysis/thickness_analysis.ipynb` | whole brain thickness models |
| `src/surface_analysis/surface_analisys.ipynb` | surface failure breakdown, adds `wm_surface_area` |
| `src/surface_analysis/surface_analisys_summary.ipynb` | WM surface area vs GA plots |
| `src/regional_thickness_demo.ipynb` | per vertex thickness on the fetal21 atlas, needs cluster access |

Legacy and superseded: `src/multivariate_analysis/`, `src/cnr_analisys/`,
`src/summary_analisys/`.

## Regenerating the data (cluster only)

```bash
# 1. Segmentations. Needs Milton's conda env, does not run on el-jobo.
source ~/miniconda3/bin/activate /neuro/labs/grantlab/research/MRI_processing/milton.candela/code/brain_age/conda_env/original_env/
cd src/processing
python3 run_SP_prediction.py --subjects ../../data/subject.csv

# 2. Surfaces. Needs a node with apptainer (hanyang) and GNU parallel.
source /neuro/users/mri.team/packages/env_MRI_team
uv run python run_SP_batch.py --subjects ../../data/subject.csv --max_workers 16
uv run python generate_native_surfaces.py

# 3. Audits
uv run python surface_audit.py --batch      # data/surface_audit.csv
uv run python thickness_audit.py            # data/thickness_audit.csv

# 4. Image metrics
cd ..
uv run python image_quality_metrics.py \
    --subjects ../data/subject.csv \
    --base_path /neuro/labs/grantlab/research/MRI_processing/seungyoon.jeong/2025/Reliability/TEST/ \
    --pair_diffs

# 5. Boundary CNR, must run as a module from the repo root
cd ..
uv run python -m src.functions.CNR

# 6. Per subject Dice and Jaccard reports
cd src/reliability
uv run python batch_process.py
```

General MRI environment when needed:

```bash
micromamba activate -p /neuro/labs/grantlab/research/MRI_processing/environment
```

Set `--max_workers` to about `nproc / 4`. Failures land in
`src/processing/failed.csv` and in
`{subject}/{session}/{split}/{subject}_{session}_{split}_error.log`.

`--force_rerun` deletes `default_surfaces/*`, including the native surfaces and
thickness files, so rerun `generate_native_surfaces.py` after using it.

## Extra notes

- Run every notebook and script from the directory it lives in. Paths and imports
  are relative.
- `src/functions/CNR.py` is the exception, run it as `uv run python -m src.functions.CNR`
  from the repo root.
- These scripts rewrite CSVs in place, so keep the pipeline order:

| Writer | Rewrites |
|---|---|
| `src/functions/CNR.py` | `image_quality_metrics.csv`, `split_comparision_data.csv` |
| `src/surface_analysis/surface_analisys.ipynb` | `image_quality_metrics.csv` |
| `src/reliability/Reliability.ipynb` | `cross_split_metrics.csv`, `image_quality_metrics.csv` |
| `src/bootstrap_covariates.ipynb` | `assets/reliability_summary_table.csv` |

- `--pair_diffs` only replaces the volume columns it owns, so `cnr_*` and
  `wm_surface_area_*` survive a rerun.
- Step 2 needs `<subject>_<session>_nuc_deep_subplate_dilate_mc.nii` in each split's
  `segmentations/`. There is no fallback name, the split is skipped if it is missing.
