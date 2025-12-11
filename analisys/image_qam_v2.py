#!/usr/bin/env python3
"""
Image Quality Metrics for Fetal Brain Segmentation Reliability Analysis - v2

Key changes from v1:
- Computes SNR for ALL splits (S1, S2, S3, S4), not just S1
- Calculates SNR differences between independent split pairs (S1-S2, S3-S4)
- Removes CNR (inconclusive due to T2 harmonization)
- Adds voxel counts per split for direct comparison

Usage:
    from image_quality_metrics_v2 import compute_all_splits_quality_metrics

    quality_df = compute_all_splits_quality_metrics(subjects_df, base_path)
"""

import nibabel as nib
import numpy as np
import pandas as pd
import os


# ============================================================
# Core SNR Function
# ============================================================


def compute_snr(t2_data, mask_data, tissue_labels):
    """
    Compute Signal-to-Noise Ratio for a tissue type.

    SNR = mean(signal) / std(signal)

    Parameters:
    -----------
    t2_data : np.ndarray
        T2 image volume
    mask_data : np.ndarray
        Segmentation mask
    tissue_labels : int or list
        Label(s) for the tissue of interest

    Returns:
    --------
    snr : float
        Signal-to-noise ratio
    """
    if isinstance(tissue_labels, int):
        tissue_labels = [tissue_labels]

    tissue_mask = np.isin(mask_data, tissue_labels)
    tissue_intensities = t2_data[tissue_mask]

    if len(tissue_intensities) == 0:
        return np.nan

    mean_signal = np.mean(tissue_intensities)
    std_signal = np.std(tissue_intensities)

    if std_signal == 0:
        return np.nan

    return mean_signal / std_signal


def compute_voxel_count(mask_data, tissue_labels):
    """
    Count voxels for a tissue type.

    Parameters:
    -----------
    mask_data : np.ndarray
        Segmentation mask
    tissue_labels : int or list
        Label(s) for the tissue of interest

    Returns:
    --------
    count : int
        Number of voxels
    """
    if isinstance(tissue_labels, int):
        tissue_labels = [tissue_labels]

    tissue_mask = np.isin(mask_data, tissue_labels)
    return int(np.sum(tissue_mask))


# ============================================================
# Single Split Processing
# ============================================================


def compute_split_metrics(t2_path, seg_path, split_name):
    """
    Compute SNR and voxel counts for a single split.

    Returns:
    --------
    dict with SNR and voxel counts for each tissue type
    """
    try:
        t2_data = nib.load(t2_path).get_fdata()
        seg_data = nib.load(seg_path).get_fdata()
    except Exception as e:
        print(f"  Error loading {split_name}: {e}")
        return None

    # Define tissue labels (SP model)
    SP_LABELS = [4, 5]  # Left/Right Subplate
    CP_LABELS = [1, 42]  # Left/Right Cortical Plate
    INNER_LABELS = [160, 161]  # Left/Right Inner region

    return {
        f"snr_sp_{split_name}": compute_snr(t2_data, seg_data, SP_LABELS),
        f"snr_cp_{split_name}": compute_snr(t2_data, seg_data, CP_LABELS),
        f"snr_inner_{split_name}": compute_snr(t2_data, seg_data, INNER_LABELS),
        f"voxels_sp_{split_name}": compute_voxel_count(seg_data, SP_LABELS),
        f"voxels_cp_{split_name}": compute_voxel_count(seg_data, CP_LABELS),
        f"voxels_inner_{split_name}": compute_voxel_count(seg_data, INNER_LABELS),
    }


# ============================================================
# Subject Processing (All Splits)
# ============================================================


def compute_subject_all_splits(base_path, subject_id, session_id):
    """
    Compute SNR and voxel counts for all 4 splits of a subject.
    Also computes differences between independent split pairs.

    Parameters:
    -----------
    base_path : str
        Base path to data
    subject_id : str
        Subject identifier
    session_id : str
        Session identifier

    Returns:
    --------
    dict with all metrics for all splits plus derived metrics
    """
    splits = ["S1", "S2", "S3", "S4"]

    result = {
        "subject_id": subject_id,
        "session_id": session_id,
    }

    all_split_data = {}

    for split in splits:
        current_path = os.path.join(base_path, str(subject_id), str(session_id), split)
        t2_path = os.path.join(current_path, "recon_segmentation/recon_to31_nuc.nii")
        seg_path = os.path.join(
            current_path,
            f"segmentations/{subject_id}_{session_id}_nuc_deep_subplate_dilate_mc.nii",
        )

        if not os.path.exists(t2_path) or not os.path.exists(seg_path):
            print(f"  Missing files for {split}")
            continue

        split_metrics = compute_split_metrics(t2_path, seg_path, split)
        if split_metrics:
            all_split_data[split] = split_metrics
            result.update(split_metrics)

    # Compute differences between independent split pairs
    # S1-S2 are independent (no shared slices)
    # S3-S4 are independent (no shared slices)

    if "S1" in all_split_data and "S2" in all_split_data:
        # SNR differences (absolute)
        result["snr_sp_diff_S1S2"] = abs(
            all_split_data["S1"]["snr_sp_S1"] - all_split_data["S2"]["snr_sp_S2"]
        )
        result["snr_cp_diff_S1S2"] = abs(
            all_split_data["S1"]["snr_cp_S1"] - all_split_data["S2"]["snr_cp_S2"]
        )

        # Voxel count differences (absolute and relative)
        vox_sp_s1 = all_split_data["S1"]["voxels_sp_S1"]
        vox_sp_s2 = all_split_data["S2"]["voxels_sp_S2"]
        result["voxels_sp_diff_S1S2"] = abs(vox_sp_s1 - vox_sp_s2)
        result["voxels_sp_reldiff_S1S2"] = (
            abs(vox_sp_s1 - vox_sp_s2) / ((vox_sp_s1 + vox_sp_s2) / 2) * 100
            if (vox_sp_s1 + vox_sp_s2) > 0
            else 0
        )

        vox_cp_s1 = all_split_data["S1"]["voxels_cp_S1"]
        vox_cp_s2 = all_split_data["S2"]["voxels_cp_S2"]
        result["voxels_cp_diff_S1S2"] = abs(vox_cp_s1 - vox_cp_s2)
        result["voxels_cp_reldiff_S1S2"] = (
            abs(vox_cp_s1 - vox_cp_s2) / ((vox_cp_s1 + vox_cp_s2) / 2) * 100
            if (vox_cp_s1 + vox_cp_s2) > 0
            else 0
        )

    if "S3" in all_split_data and "S4" in all_split_data:
        # SNR differences
        result["snr_sp_diff_S3S4"] = abs(
            all_split_data["S3"]["snr_sp_S3"] - all_split_data["S4"]["snr_sp_S4"]
        )
        result["snr_cp_diff_S3S4"] = abs(
            all_split_data["S3"]["snr_cp_S3"] - all_split_data["S4"]["snr_cp_S4"]
        )

        # Voxel count differences
        vox_sp_s3 = all_split_data["S3"]["voxels_sp_S3"]
        vox_sp_s4 = all_split_data["S4"]["voxels_sp_S4"]
        result["voxels_sp_diff_S3S4"] = abs(vox_sp_s3 - vox_sp_s4)
        result["voxels_sp_reldiff_S3S4"] = (
            abs(vox_sp_s3 - vox_sp_s4) / ((vox_sp_s3 + vox_sp_s4) / 2) * 100
            if (vox_sp_s3 + vox_sp_s4) > 0
            else 0
        )

        vox_cp_s3 = all_split_data["S3"]["voxels_cp_S3"]
        vox_cp_s4 = all_split_data["S4"]["voxels_cp_S4"]
        result["voxels_cp_diff_S3S4"] = abs(vox_cp_s3 - vox_cp_s4)
        result["voxels_cp_reldiff_S3S4"] = (
            abs(vox_cp_s3 - vox_cp_s4) / ((vox_cp_s3 + vox_cp_s4) / 2) * 100
            if (vox_cp_s3 + vox_cp_s4) > 0
            else 0
        )

    # Mean SNR across all splits
    snr_sp_values = [
        all_split_data[s][f"snr_sp_{s}"]
        for s in all_split_data
        if f"snr_sp_{s}" in all_split_data[s]
    ]
    snr_cp_values = [
        all_split_data[s][f"snr_cp_{s}"]
        for s in all_split_data
        if f"snr_cp_{s}" in all_split_data[s]
    ]

    result["snr_sp_mean"] = np.mean(snr_sp_values) if snr_sp_values else np.nan
    result["snr_cp_mean"] = np.mean(snr_cp_values) if snr_cp_values else np.nan
    result["snr_sp_std"] = np.std(snr_sp_values) if len(snr_sp_values) > 1 else np.nan
    result["snr_cp_std"] = np.std(snr_cp_values) if len(snr_cp_values) > 1 else np.nan

    return result


# ============================================================
# Batch Processing
# ============================================================


def compute_all_splits_quality_metrics(
    subjects_df, base_path, output_csv="image_quality_metrics.csv"
):
    """
    Compute quality metrics for all subjects across all splits.

    Parameters:
    -----------
    subjects_df : pd.DataFrame
        DataFrame with subject_id and session_id columns
    base_path : str
        Base path to data directory
    output_csv : str
        Path to save output CSV

    Returns:
    --------
    pd.DataFrame with all quality metrics
    """
    all_metrics = []

    for idx, row in subjects_df.iterrows():
        subject_id = str(row["subject_id"])
        session_id = str(row["session_id"])

        print(f"[{idx+1}/{len(subjects_df)}] Processing {subject_id}...")

        metrics = compute_subject_all_splits(base_path, subject_id, session_id)
        all_metrics.append(metrics)

    df = pd.DataFrame(all_metrics)

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\nSaved quality metrics to: {output_csv}")

    return df


# ============================================================
# Summary Functions
# ============================================================


def print_snr_summary(quality_df):
    """Print a summary of SNR metrics."""
    print("\n" + "=" * 60)
    print("SNR METRICS SUMMARY")
    print("=" * 60)

    print(f"\nN subjects: {len(quality_df)}\n")

    # Per-split SNR
    print("SNR by Split (Subplate):")
    for split in ["S1", "S2", "S3", "S4"]:
        col = f"snr_sp_{split}"
        if col in quality_df.columns:
            data = quality_df[col].dropna()
            print(f"  {split}: {data.mean():.2f} ± {data.std():.2f}")

    print("\nSNR by Split (Cortical Plate):")
    for split in ["S1", "S2", "S3", "S4"]:
        col = f"snr_cp_{split}"
        if col in quality_df.columns:
            data = quality_df[col].dropna()
            print(f"  {split}: {data.mean():.2f} ± {data.std():.2f}")

    # SNR differences between independent pairs
    print("\nSNR Differences (Independent Split Pairs):")
    for pair in ["S1S2", "S3S4"]:
        col = f"snr_sp_diff_{pair}"
        if col in quality_df.columns:
            data = quality_df[col].dropna()
            print(f"  Subplate {pair}: {data.mean():.2f} ± {data.std():.2f}")

    print("\n" + "=" * 60)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import sys

    BASE_PATH = "/neuro/labs/grantlab/research/MRI_processing/seungyoon.jeong/2025/Reliability/TEST/"
    SUBJECTS_CSV = "subject.csv"

    if not os.path.exists(SUBJECTS_CSV):
        print(f"Error: {SUBJECTS_CSV} not found.")
        sys.exit(1)

    subjects_df = pd.read_csv(SUBJECTS_CSV)
    print(f"Found {len(subjects_df)} subjects\n")

    quality_df = compute_all_splits_quality_metrics(
        subjects_df, BASE_PATH, output_csv="image_quality_metrics.csv"
    )

    print_snr_summary(quality_df)
    print("\nDone!")
