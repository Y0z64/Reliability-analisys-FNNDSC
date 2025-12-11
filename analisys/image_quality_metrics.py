#!/usr/bin/env python3
"""
Image Quality Metrics for Fetal Brain Segmentation Reliability Analysis

This module computes SNR and CNR from T2 images using segmentation masks,
and correlates these with segmentation reliability metrics.

Usage:
    # In your Reliability.py notebook, add:
    from image_quality_metrics import compute_image_quality_metrics, plot_quality_vs_reliability

    # Compute metrics for all subjects
    quality_df = compute_image_quality_metrics(subjects_df, base_path)

    # Plot correlation with reliability
    plot_quality_vs_reliability(quality_df, reliability_df)
"""

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os


# ============================================================
# Core SNR/CNR Functions
# ============================================================


def compute_snr(t2_data, mask_data, tissue_label):
    """
    Compute Signal-to-Noise Ratio for a tissue type.

    SNR = mean(signal) / std(signal)

    This uses the standard deviation within the tissue as a noise proxy,
    which is appropriate when no background region is available.

    Parameters:
    -----------
    t2_data : np.ndarray
        T2 image volume
    mask_data : np.ndarray
        Segmentation mask
    tissue_label : int or list
        Label(s) for the tissue of interest

    Returns:
    --------
    snr : float
        Signal-to-noise ratio
    """
    if isinstance(tissue_label, int):
        tissue_label = [tissue_label]

    tissue_mask = np.isin(mask_data, tissue_label)
    tissue_intensities = t2_data[tissue_mask]

    if len(tissue_intensities) == 0:
        return np.nan

    mean_signal = np.mean(tissue_intensities)
    std_signal = np.std(tissue_intensities)

    if std_signal == 0:
        return np.nan

    return mean_signal / std_signal


def compute_cnr(
    t2_data, mask_data, tissue_a_labels, tissue_b_labels, noise_labels=None
):
    """
    Compute Contrast-to-Noise Ratio between two tissue types.

    CNR = |mean(A) - mean(B)| / noise_estimate

    Parameters:
    -----------
    t2_data : np.ndarray
        T2 image volume
    mask_data : np.ndarray
        Segmentation mask
    tissue_a_labels : int or list
        Label(s) for tissue A (e.g., subplate)
    tissue_b_labels : int or list
        Label(s) for tissue B (e.g., cortical plate)
    noise_labels : int or list, optional
        Label(s) to use for noise estimation. If None, uses tissue A.

    Returns:
    --------
    cnr : float
        Contrast-to-noise ratio
    """
    if isinstance(tissue_a_labels, int):
        tissue_a_labels = [tissue_a_labels]
    if isinstance(tissue_b_labels, int):
        tissue_b_labels = [tissue_b_labels]

    mask_a = np.isin(mask_data, tissue_a_labels)
    mask_b = np.isin(mask_data, tissue_b_labels)

    intensities_a = t2_data[mask_a]
    intensities_b = t2_data[mask_b]

    if len(intensities_a) == 0 or len(intensities_b) == 0:
        return np.nan

    mean_a = np.mean(intensities_a)
    mean_b = np.mean(intensities_b)

    # Noise estimation
    if noise_labels is not None:
        if isinstance(noise_labels, int):
            noise_labels = [noise_labels]
        noise_mask = np.isin(mask_data, noise_labels)
        noise_intensities = t2_data[noise_mask]
        if len(noise_intensities) == 0:
            noise_std = np.std(intensities_a)
        else:
            noise_std = np.std(noise_intensities)
    else:
        # Use pooled standard deviation of both tissues
        noise_std = np.sqrt((np.var(intensities_a) + np.var(intensities_b)) / 2)

    if noise_std == 0:
        return np.nan

    return np.abs(mean_a - mean_b) / noise_std


def compute_tissue_stats(t2_data, mask_data, labels):
    """
    Compute basic statistics for a tissue region.

    Returns:
    --------
    dict with mean, std, min, max, volume (voxel count)
    """
    if isinstance(labels, int):
        labels = [labels]

    mask = np.isin(mask_data, labels)
    intensities = t2_data[mask]

    if len(intensities) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "volume_voxels": 0,
        }

    return {
        "mean": np.mean(intensities),
        "std": np.std(intensities),
        "min": np.min(intensities),
        "max": np.max(intensities),
        "volume_voxels": len(intensities),
    }


# ============================================================
# Subject Processing Functions
# ============================================================


def compute_subject_quality_metrics(
    t2_path, seg_path, subject_id, session_id, split="S1"
):
    """
    Compute all image quality metrics for a single subject/split.

    Parameters:
    -----------
    t2_path : str
        Path to T2 image (recon_to31_nuc.nii)
    seg_path : str
        Path to segmentation mask
    subject_id : str
        Subject identifier
    session_id : str
        Session identifier
    split : str
        Split identifier (default 'S1')

    Returns:
    --------
    dict with all computed metrics
    """
    # Load data
    try:
        t2_data = nib.load(t2_path).get_fdata()
        seg_data = nib.load(seg_path).get_fdata()
    except Exception as e:
        print(f"Error loading {subject_id}/{session_id}/{split}: {e}")
        return None

    # Define tissue labels (SP model)
    SP_LABELS = [4, 5]  # Left/Right Subplate
    CP_LABELS = [1, 42]  # Left/Right Cortical Plate
    INNER_LABELS = [160, 161]  # Left/Right Inner region

    # Compute SNR for each tissue
    snr_sp = compute_snr(t2_data, seg_data, SP_LABELS)
    snr_cp = compute_snr(t2_data, seg_data, CP_LABELS)
    snr_inner = compute_snr(t2_data, seg_data, INNER_LABELS)

    # Compute CNR between tissue pairs
    cnr_sp_cp = compute_cnr(t2_data, seg_data, SP_LABELS, CP_LABELS)
    cnr_sp_inner = compute_cnr(t2_data, seg_data, SP_LABELS, INNER_LABELS)
    cnr_cp_inner = compute_cnr(t2_data, seg_data, CP_LABELS, INNER_LABELS)

    # Get tissue statistics
    stats_sp = compute_tissue_stats(t2_data, seg_data, SP_LABELS)
    stats_cp = compute_tissue_stats(t2_data, seg_data, CP_LABELS)
    stats_inner = compute_tissue_stats(t2_data, seg_data, INNER_LABELS)

    return {
        "subject_id": subject_id,
        "session_id": session_id,
        "split": split,
        # SNR
        "snr_subplate": snr_sp,
        "snr_cortical_plate": snr_cp,
        "snr_inner": snr_inner,
        # CNR
        "cnr_sp_cp": cnr_sp_cp,
        "cnr_sp_inner": cnr_sp_inner,
        "cnr_cp_inner": cnr_cp_inner,
        # Tissue stats - Subplate
        "sp_mean_intensity": stats_sp["mean"],
        "sp_std_intensity": stats_sp["std"],
        "sp_volume_voxels": stats_sp["volume_voxels"],
        # Tissue stats - Cortical Plate
        "cp_mean_intensity": stats_cp["mean"],
        "cp_std_intensity": stats_cp["std"],
        "cp_volume_voxels": stats_cp["volume_voxels"],
        # Tissue stats - Inner
        "inner_mean_intensity": stats_inner["mean"],
        "inner_std_intensity": stats_inner["std"],
        "inner_volume_voxels": stats_inner["volume_voxels"],
    }


def compute_image_quality_metrics(
    subjects_df, base_path, split="S1", output_csv="image_quality_metrics.csv"
):
    """
    Compute image quality metrics for all subjects.

    Parameters:
    -----------
    subjects_df : pd.DataFrame
        DataFrame with 'subject_id' and 'session_id' columns
    base_path : str
        Base path to data directory
    split : str
        Which split to use for computing metrics (default 'S1')
    output_csv : str
        Path to save results CSV

    Returns:
    --------
    pd.DataFrame with quality metrics for all subjects
    """
    results = []

    for idx, row in subjects_df.iterrows():
        subject_id = str(row["subject_id"])
        session_id = str(row["session_id"])

        print(f"Processing {subject_id} ({idx + 1}/{len(subjects_df)})...")

        # Build paths
        split_path = os.path.join(base_path, subject_id, session_id, split)
        t2_path = os.path.join(split_path, "recon_segmentation", "recon_to31_nuc.nii")
        seg_path = os.path.join(
            split_path,
            "segmentations",
            f"{subject_id}_{session_id}_nuc_deep_subplate_dilate_mc.nii",
        )

        # Check if files exist
        if not os.path.exists(t2_path):
            print(f"  Warning: T2 not found: {t2_path}")
            continue
        if not os.path.exists(seg_path):
            print(f"  Warning: Segmentation not found: {seg_path}")
            continue

        # Compute metrics
        metrics = compute_subject_quality_metrics(
            t2_path, seg_path, subject_id, session_id, split
        )

        if metrics is not None:
            results.append(metrics)
            print(
                f"  CNR(SP-CP): {metrics['cnr_sp_cp']:.2f}, SNR(SP): {metrics['snr_subplate']:.2f}"
            )

    # Create DataFrame
    quality_df = pd.DataFrame(results)

    # Save to CSV
    if output_csv:
        quality_df.to_csv(output_csv, index=False)
        print(f"\nSaved quality metrics to: {output_csv}")

    return quality_df


# ============================================================
# Merge with Reliability Metrics
# ============================================================


def merge_quality_and_reliability(
    quality_csv, reliability_csv, output_csv="combined_metrics.csv"
):
    """
    Merge image quality metrics with reliability metrics (Dice, etc.)

    Parameters:
    -----------
    quality_csv : str
        Path to image_quality_metrics.csv
    reliability_csv : str
        Path to cross_split_metrics.csv
    output_csv : str
        Path to save combined CSV

    Returns:
    --------
    """
    quality_df = pd.read_csv(quality_csv)
    reliability_df = pd.read_csv(reliability_csv)

    # Compute mean Dice per subject per model
    reliability_summary = (
        reliability_df.groupby(["subject_id", "session_id", "model"])
        .agg(
            {
                "dice": ["mean", "std"],
                "jaccard": ["mean", "std"],
                "relative_diff": ["mean", "std"],
            }
        )
        .reset_index()
    )

    # Flatten column names
    reliability_summary.columns = [
        "_".join(col).strip("_") for col in reliability_summary.columns
    ]

    # Merge
    combined_df = quality_df.merge(
        reliability_summary, on=["subject_id", "session_id"], how="left"
    )


    return combined_df


# ============================================================
# Plotting Functions
# ============================================================


def plot_quality_vs_reliability(
    combined_df,
    model_filter="SP Model - Subplate",
    quality_metric="cnr_sp_cp",
    reliability_metric="dice_mean",
    output_path=None,
):
    """
    Create scatter plot of image quality vs segmentation reliability.

    Parameters:
    -----------
    combined_df : pd.DataFrame
        DataFrame with both quality and reliability metrics
    model_filter : str
        Which model's reliability to plot
    quality_metric : str
        Column name for quality metric (x-axis)
    reliability_metric : str
        Column name for reliability metric (y-axis)
    output_path : str, optional
        Path to save figure

    Returns:
    --------
    matplotlib Figure
    """
    # Filter to specific model
    if "model" in combined_df.columns:
        plot_df = combined_df[combined_df["model"] == model_filter].copy()
    else:
        plot_df = combined_df.copy()

    # Remove NaN
    plot_df = plot_df.dropna(subset=[quality_metric, reliability_metric])

    if len(plot_df) < 3:
        print(
            f"Warning: Only {len(plot_df)} valid data points. Need at least 3 for correlation."
        )
        return None

    fig, ax = plt.subplots(figsize=(10, 7))

    # Scatter plot
    ax.scatter(
        plot_df[quality_metric],
        plot_df[reliability_metric],
        s=80,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )

    # Add subject labels
    for _, row in plot_df.iterrows():
        ax.annotate(
            row["subject_id"],
            (row[quality_metric], row[reliability_metric]),
            fontsize=8,
            alpha=0.7,
            xytext=(5, 5),
            textcoords="offset points",
        )

    # Compute correlation
    r, p_value = stats.pearsonr(plot_df[quality_metric], plot_df[reliability_metric])

    # Add regression line
    z = np.polyfit(plot_df[quality_metric], plot_df[reliability_metric], 1)
    p = np.poly1d(z)
    x_line = np.linspace(
        plot_df[quality_metric].min(), plot_df[quality_metric].max(), 100
    )
    ax.plot(
        x_line,
        p(x_line),
        "r--",
        alpha=0.8,
        linewidth=2,
        label=f"r = {r:.3f} (p = {p_value:.3f})",
    )

    # Labels and formatting
    ax.set_xlabel(quality_metric.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(reliability_metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(
        f"Image Quality vs Segmentation Reliability\n{model_filter}",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")

    return fig

def plot_snr_cnr_comparison(quality_df):
    """
    Create side-by-side bar plots comparing SNR and CNR across tissue types.
    
    Parameters:
    -----------
    quality_df : pd.DataFrame
        DataFrame with quality metrics
    
    Returns:
    --------
    matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # === Left plot: SNR comparison ===
    snr_metrics = ['snr_subplate', 'snr_cortical_plate', 'snr_inner']
    snr_labels = ['Subplate', 'Cortical Plate', 'Inner Region']
    
    snr_means = [quality_df[m].mean() for m in snr_metrics]
    snr_stds = [quality_df[m].std() for m in snr_metrics]
    
    x1 = np.arange(len(snr_labels))
    bars1 = ax1.bar(x1, snr_means, yerr=snr_stds, capsize=5,
                    color=['#e74c3c', '#3498db', '#2ecc71'],
                    edgecolor='black', alpha=0.8)
    
    ax1.set_xticks(x1)
    ax1.set_xticklabels(snr_labels, fontsize=11)
    ax1.set_ylabel('Signal-to-Noise Ratio (SNR)', fontsize=12)
    ax1.set_title('SNR by Tissue Type\n(Mean ± SD across subjects)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, mean in zip(bars1, snr_means):
        ax1.annotate(f'{mean:.2f}',
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 5), textcoords='offset points',
                     ha='center', fontsize=11, fontweight='bold')
    
    # === Right plot: CNR comparison ===
    cnr_metrics = ['cnr_sp_cp', 'cnr_sp_inner', 'cnr_cp_inner']
    cnr_labels = ['Subplate vs\nCortical Plate', 'Subplate vs\nInner Region', 'Cortical Plate vs\nInner Region']
    
    cnr_means = [quality_df[m].mean() for m in cnr_metrics]
    cnr_stds = [quality_df[m].std() for m in cnr_metrics]
    
    x2 = np.arange(len(cnr_labels))
    bars2 = ax2.bar(x2, cnr_means, yerr=cnr_stds, capsize=5,
                    color=['#9b59b6', '#f39c12', '#1abc9c'],
                    edgecolor='black', alpha=0.8)
    
    ax2.set_xticks(x2)
    ax2.set_xticklabels(cnr_labels, fontsize=11)
    ax2.set_ylabel('Contrast-to-Noise Ratio (CNR)', fontsize=12)
    ax2.set_title('CNR Between Tissue Boundaries\n(Mean ± SD across subjects)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, mean in zip(bars2, cnr_means):
        ax2.annotate(f'{mean:.2f}',
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 5), textcoords='offset points',
                     ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig
# ============================================================
# Summary Report Function
# ============================================================


def print_quality_summary(quality_df):
    """Print a summary of image quality metrics."""
    print("\n" + "=" * 60)
    print("IMAGE QUALITY METRICS SUMMARY")
    print("=" * 60)

    metrics = {
        "CNR (Subplate vs Cortical Plate)": "cnr_sp_cp",
        "CNR (Subplate vs Inner)": "cnr_sp_inner",
        "CNR (Cortical Plate vs Inner)": "cnr_cp_inner",
        "SNR (Subplate)": "snr_subplate",
        "SNR (Cortical Plate)": "snr_cortical_plate",
        "SNR (Inner Region)": "snr_inner",
    }

    print(f"\nN subjects: {len(quality_df)}\n")
    print(f"{'Metric':<35} {'Mean':>10} {'SD':>10} {'Min':>10} {'Max':>10}")
    print("-" * 75)

    for name, col in metrics.items():
        data = quality_df[col].dropna()
        print(
            f"{name:<35} {data.mean():>10.2f} {data.std():>10.2f} {data.min():>10.2f} {data.max():>10.2f}"
        )

    print("\n" + "=" * 60)


# ============================================================
# Main execution example
# ============================================================

if __name__ == "__main__":
    """
    Example usage - modify paths as needed.
    """
    import sys

    # Configuration - MODIFY THESE PATHS
    BASE_PATH = "/neuro/labs/grantlab/research/MRI_processing/seungyoon.jeong/2025/Reliability/TEST/"
    SUBJECTS_CSV = "subjects_batched/test.csv"
    RELIABILITY_CSV = "cross_split_metrics.csv"

    # Check if files exist
    if not os.path.exists(SUBJECTS_CSV):
        print(f"Error: {SUBJECTS_CSV} not found.")
        print("Run this script from the directory containing your subject.csv")
        sys.exit(1)

    # Load subjects
    subjects_df = pd.read_csv(SUBJECTS_CSV)
    print(f"Found {len(subjects_df)} subjects\n")

    # Compute quality metrics
    quality_df = compute_image_quality_metrics(
        subjects_df, BASE_PATH, split="S1", output_csv="image_quality_metrics.csv"
    )

    # Print summary
    print_quality_summary(quality_df)

    # Generate distribution plots
    plot_snr_cnr_comparison(quality_df, output_path="cnr_comparison.png")

    # If reliability metrics exist, merge and plot correlation
    if os.path.exists(RELIABILITY_CSV):
        combined_df = merge_quality_and_reliability(
            "image_quality_metrics.csv",
            RELIABILITY_CSV,
            output_csv="combined_metrics.csv",
        )

        # Plot for each model
        for model in [
            "SP Model - Subplate",
            "SP Model - Cortical Plate",
            "5-Label Model - Cortical Plate",
        ]:
            safe_name = model.replace(" ", "_").replace("-", "_").lower()
            plot_quality_vs_reliability(
                combined_df,
                model_filter=model,
                quality_metric="cnr_sp_cp",
                reliability_metric="dice_mean",
                output_path=f"quality_vs_reliability_{safe_name}.png",
            )

    print("\nDone!")
