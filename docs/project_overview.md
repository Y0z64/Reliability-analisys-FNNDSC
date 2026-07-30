# Project overview

Measures the reliability (reproducibility) of a fetal brain **subplate (SP)**
segmentation model, replicating a prior lab study that measured the same thing for
the cortical plate (CP) model. That paper is in `assets/reliability_paper.pdf`,
its figures in `assets/relability_paper_figures.pdf`.

## Design

One MRI scan per subject is reconstructed **four times** from different slice
subsets (`S1 S2 S3 S4`). All four reconstruct the same brain, so a reliable model
should produce identical outputs; any difference measures model instability.

- `S1-S2` and `S3-S4` are the two **non-overlapping** pairs — these are the pairs
  every model is fit on.
- The other combinations (S1-S3, S1-S4, S2-S3, S2-S4) share slices and are only
  used for secondary checks.

## Sample

- N = 21 subjects in `data/subject.csv`, 46 usable split pairs.
- GA 22–32 weeks, plus a few above 32. The SP model is optimised for < 32 weeks;
  `thickness_audit.py` filters to `GA <= 32`.
- Stack counts 3–11 per subject.
- Thickness models use a smaller subset (13 subjects/pair) — the reason
  `paper_models_thickness_subjects.ipynb` exists.

## What gets measured

| Output | Reliability metric | Where |
|---|---|---|
| SP volume | `abs_diff_sp`, `rel_diff_sp` between paired splits | `split_comparision_data.csv` |
| WM surface area | `wm_surface_area_diff` | `split_comparision_data.csv` |
| SP thickness | `thk_diff`, `apd_thk` | `thickness_audit.csv` |
| Segmentation overlap | Dice, Jaccard, relative volume difference | `cross_split_metrics.csv` |

Covariates: gestational age (`GA`), manual quality score (`qa_mean`), stack count,
SNR and CNR.

## Repo

- GitHub: `github.com/Y0z64/Reliability-analisys-FNNDSC`
- Main branch `new_processing`; this work is on `restructuring`.
- Contacts: Yair Beltran, Andrea Gondova, Seungyoon Jeong.

## Where to look

| | |
|---|---|
| How to run anything | `docs/pipeline.md` |
| What a CSV column means and what wrote it | `docs/data_dictionary.md` |
| Envs and cluster paths | `docs/environment.md` |
| Traps before you change anything | `docs/handoff_notes.md` |
