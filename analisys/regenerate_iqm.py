import pandas as pd
import matplotlib.pyplot as plt
import os
from image_quality_metrics import (
    compute_subject_quality_metrics,
)

# 1. Configuration
# base_path = "/home/yair/Projects/Tests"
base_path = "/neuro/labs/grantlab/research/MRI_processing/seungyoon.jeong/2025/Reliability/TEST/"
# subject_id = "FCB145"
# session_id = "2019.03.20-032Y-MR_EI_Fetal_Neuro-29197"

CSV_PATH = "../data/subject.csv"

subjects_df = pd.read_csv(CSV_PATH)
# print(f"Found {len(subjects_df)} subjects to process")


def iqm(base_path, subject_id, session_id, split_to_use):
    # Compute quality metrics for the current subject (using S1)
    print(f"\n{'='*60}\nComputing Image Quality Metrics\n{'='*60}")
    current_path = os.path.join(base_path, subject_id, session_id, split_to_use)
    t2_path = os.path.join(current_path, "recon_segmentation/recon_to31_nuc.nii")
    seg_path = os.path.join(
        current_path,
        f"segmentations/{subject_id}_{session_id}_nuc_deep_subplate_dilate_mc.nii",
    )
    scaling_inv_path = os.path.join(
        current_path, "recon_segmentation/alignment_temp/recon_to31_inv.xfm"
    )

    # Compute metrics for this subject
    quality_metrics = compute_subject_quality_metrics(
        t2_path, seg_path, subject_id, session_id, scaling_inv_path, split_to_use
    )

    if quality_metrics:
        print(f"\nImage Quality Metrics for {subject_id} ({split_to_use}):")
        print(f"  SNR (Subplate):        {quality_metrics['snr_subplate']:.2f}")
        print(f"  SNR (Cortical Plate):  {quality_metrics['snr_cortical_plate']:.2f}")
        print(f"  CNR (SP vs CP):        {quality_metrics['cnr_sp_cp']:.2f}")
        print(f"  CNR (SP vs Inner):     {quality_metrics['cnr_sp_inner']:.2f}")
        print(f"  CNR (CP vs Inner):     {quality_metrics['cnr_cp_inner']:.2f}")

        # Save or update image quality metrics CSV
        quality_csv = "image_quality_metrics.csv"
        quality_df_new = pd.DataFrame([quality_metrics])

        if os.path.exists(quality_csv):
            quality_df_existing = pd.read_csv(quality_csv)
            mask = ~(
                (quality_df_existing["subject_id"].astype(str) == str(subject_id))
                & (quality_df_existing["session_id"].astype(str) == str(session_id))
                & (quality_df_existing["split"].astype(str) == str(split_to_use))
            )
            quality_df_existing = quality_df_existing[mask]
            quality_df = pd.concat(
                [quality_df_existing, quality_df_new], ignore_index=True
            )
        else:
            quality_df = quality_df_new

        quality_df.to_csv(quality_csv, index=False)
        print(f"\nSaved quality metrics for {subject_id} to {quality_csv}")

        # # Print summary for all subjects in CSV
        # print_quality_summary(quality_df)

        # # Print quality metrics
        # print_volume_summary(subject_id, split, quality_metrics)

    else:
        print(f"Failed for: {subject_id}/{session_id}/{split}")


splits = ["S1", "S2", "S3", "S4"]
for idx, row in subjects_df.iterrows():
    subject_id = str(row["subject_id"])
    session_id = str(row["session_id"])

    pdf_output_path = f"../reports/{subject_id}_{session_id}_reliability_report.pdf"

    for split in splits:
        _ = iqm(base_path, subject_id, session_id, split)

    print("Finished")
