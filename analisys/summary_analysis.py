#!/usr/bin/env python
# coding: utf-8

# # Reliability Study Summary Analysis
# 
# This notebook generates summary visualizations from:
# - cross_split_metrics.csv (Dice, Jaccard, relative_diff per model/subject)
# - image_quality_metrics.csv (SNR, CNR per subject)
# 
# Run each cell in order in your Jupyter notebook.
# 
# 

# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load data and strip whitespace from column names AND string values
reliability_df = pd.read_csv("../data/cross_split_metrics.csv")

quality_df = pd.read_csv("../data/image_quality_metrics.csv")

infodump_df = pd.read_csv("../data/raw_subject_data.csv")

subjects_df = pd.read_csv("../data/subject.csv")

split_pair_diffs = pd.read_csv("../data/split_comparision_data.csv")


# In[2]:


# Cell 1a Data access functions (LEAVE COLLAPSED)
def get_split_data(quality_df, subject_id, session_id, split):
    """
    Get data for a specific subject and split.

    Parameters:
    -----------
    quality_df : pd.DataFrame
        Long-format dataframe with columns: subject_id, session_id, split, ...
    subject_id : str
        Subject identifier
    split : str
        Split name (S1, S2, S3, S4)

    Returns:
    --------
    pd.Series or None if not found
    """
    mask = (
        (quality_df["subject_id"].astype(str) == str(subject_id))
        & (quality_df["split"] == split)
        & (quality_df["session_id"] == session_id)
    )
    result = quality_df[mask]
    return result.iloc[0] if len(result) > 0 else None


def _compute_split_pair_diff(quality_df, column, split1, split2, relative=True):
    """
    Compute difference between two splits for a given column.
    Uses subject_id + session_id as unique identifier to handle subjects with multiple sessions.
    """
    # Get data for each split
    s1_df = quality_df[quality_df["split"] == split1][
        ["subject_id", "session_id", column]
    ].copy()
    s2_df = quality_df[quality_df["split"] == split2][
        ["subject_id", "session_id", column]
    ].copy()

    # Rename columns for merge
    s1_df = s1_df.rename(columns={column: "val1"})
    s2_df = s2_df.rename(columns={column: "val2"})

    # Merge on subject_id AND session_id to handle subjects with multiple sessions
    result = s1_df.merge(s2_df, on=["subject_id", "session_id"], how="inner")

    # Create a display label (subject_id only, for cleaner plot labels)
    result["label"] = result["subject_id"].astype(str)

    # Compute difference
    if relative:
        mean_val = (result["val1"] + result["val2"]) / 2
        result["diff"] = abs(result["val1"] - result["val2"]) / mean_val * 100
    else:
        result["diff"] = abs(result["val1"] - result["val2"])

    return result


def get_column_by_split(quality_df, column, split):
    """
    Get a column's values for all subjects at a specific split.

    Parameters:
    -----------
    quality_df : pd.DataFrame
        Long-format dataframe
    column : str
        Column name (e.g., 'native_vol_sp', 'snr_subplate')
    split : str
        Split name (S1, S2, S3, S4)

    Returns:
    --------
    pd.Series with subject_id as index
    """
    split_data = quality_df[quality_df["split"] == split].copy()
    split_data = split_data.set_index("subject_id")
    if column in split_data.columns:
        return split_data[column]
    return pd.Series(dtype=float)


def get_subject_across_splits(quality_df, subject_id, columns=None):
    """
    Get all split data for a single subject, pivoted to wide format.

    Parameters:
    -----------
    quality_df : pd.DataFrame
        Long-format dataframe
    subject_id : str
        Subject identifier
    columns : list or None
        Specific columns to include (None = all)

    Returns:
    --------
    dict with split names as keys, containing the row data
    """
    subject_data = quality_df[quality_df["subject_id"].astype(str) == str(subject_id)]
    result = {}
    for _, row in subject_data.iterrows():
        split = row["split"]
        if columns:
            result[split] = {col: row[col] for col in columns if col in row.index}
        else:
            result[split] = row.to_dict()
    return result


def get_unique_subjects(quality_df):
    """Get list of unique subjects (subject_id, session_id) as tuples."""
    return list(quality_df[["subject_id", "session_id"]].drop_duplicates().itertuples(index=False, name=None))


def get_available_splits(quality_df):
    """Get list of available splits in the data."""
    return sorted(quality_df["split"].unique())


# In[3]:


# Cell 1b Summary

print(f"Subjects found: {subjects_df['subject_id'].nunique()} subjects")
print(
    f"Reliability data: {len(reliability_df)} rows, {reliability_df['subject_id'].nunique()} subjects"
)
print(
    f"Quality data: {len(quality_df)} rows, {quality_df['subject_id'].nunique()} subjects"
)
print(f"  Splits available: {get_available_splits(quality_df)}")
print(f"Infodump data: {infodump_df['subject_id'].nunique()} subjects")
print(f"\nModels in reliability data: {reliability_df['model'].unique().tolist()}")

# Show quality_df columns for reference
print(f"\nQuality metrics columns:")
print(f"  {quality_df.columns.tolist()}")

print(f"\nSubjects columns :")
print(f"  {subjects_df.columns.tolist()}")


# In[4]:


# 1c. GA Distribution: Scatter Plot and Bar Plot Side by Side
def plot_ga_distribution(quality_df):
    """
    Side-by-side scatter plot (GA vs Total Native Volume) and bar plot (GA distribution).
    GA values are floored and grouped by whole numbers.

    Parameters:
    -----------
    quality_df : pd.DataFrame
        Long-format dataframe with native volume metrics and GA
    """
    quality_df = quality_df.copy()
    quality_df["subject_id"] = quality_df["subject_id"].astype(str)
    quality_df["session_id"] = quality_df["session_id"].astype(str)

    # Floor GA values to whole numbers
    quality_df["GA_floored"] = np.floor(quality_df["GA"]).astype(int)

    # Compute total native volume (sum of SP + CP + Inner)
    quality_df["total_native_vol"] = (
        (quality_df["native_vol_sp"]
        + quality_df["native_vol_cp"]
        + quality_df["native_vol_inner"]) / 1000 #<- mm^3 to cm^3
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # === Left: Scatter Plot (GA vs Total Native Volume) ===
    ax1 = axes[0]

    # Get unique subjects for color mapping
    unique_subjects = quality_df["subject_id"].unique()
    n_subjects = len(unique_subjects)
    subject_cmap = plt.cm.get_cmap("tab20", n_subjects)
    subject_colors = {subj: subject_cmap(i) for i, subj in enumerate(unique_subjects)}

    # Group by subject_id and session_id to plot 4 splits together
    grouped = quality_df.groupby(["subject_id", "session_id"])

    for (subj_id, sess_id), group in grouped:
        color = subject_colors[subj_id]
        ga_values = group["GA"].values
        vol_values = group["total_native_vol"].values

        # Plot individual points with low alpha
        ax1.scatter(
            ga_values, vol_values, c=[color], s=40, alpha=0.4, edgecolors="none"
        )

        # Plot mean as a larger marker with edge
        mean_ga = ga_values.mean()
        mean_vol = vol_values.mean()
        ax1.scatter(
            mean_ga,
            mean_vol,
            c=[color],
            s=120,
            alpha=0.9,
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
        )

        # Draw a convex hull or ellipse around the 4 splits
        if len(ga_values) >= 3:
            from matplotlib.patches import Ellipse

            # Calculate spread
            ga_std = ga_values.std() if ga_values.std() > 0 else 0.1
            vol_std = vol_values.std() if vol_values.std() > 0 else 100

            ellipse = Ellipse(
                (mean_ga, mean_vol),
                width=ga_std * 4,  # Cover ~2 std on each side
                height=vol_std * 4,
                alpha=0.15,
                facecolor=color,
                edgecolor=color,
                linewidth=1,
            )
            ax1.add_patch(ellipse)

        # Add subject label near mean
        ax1.annotate(
            subj_id[:6],
            (mean_ga, mean_vol),
            fontsize=6,
            alpha=0.7,
            xytext=(3, 3),
            textcoords="offset points",
        )

    ax1.set_xlabel("Gestational Age (weeks)", fontsize=12)
    ax1.set_ylabel("Total Native Volume (cm³)", fontsize=12)
    ax1.set_title(
        "GA vs Total Brain Volume\n(Large dots = subject mean, ellipses = split spread)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)

    # Add correlation using subject means
    subject_means = (
        quality_df.groupby(["subject_id", "session_id"])
        .agg({"GA": "mean", "total_native_vol": "mean"})
        .dropna()
    )

    if len(subject_means) >= 3:
        r, p = stats.pearsonr(subject_means["GA"], subject_means["total_native_vol"])
        ax1.text(
            0.05,
            0.95,
            f"r = {r:.3f}\np = {p:.3f}",
            transform=ax1.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # Add regression line
        z = np.polyfit(subject_means["GA"], subject_means["total_native_vol"], 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(subject_means["GA"].min(), subject_means["GA"].max(), 100)
        ax1.plot(
            x_line, p_line(x_line), "r--", alpha=0.8, linewidth=2, label="Linear fit"
        )
        ax1.legend(loc="lower right")

    # === Right: Bar Plot (GA Distribution) ===
    ax2 = axes[1]

    # Get unique subject/session combinations for GA distribution
    ga_unique = quality_df.groupby(["subject_id", "session_id"])["GA_floored"].first()
    ga_counts = ga_unique.value_counts().sort_index()
    ga_sorted = ga_counts.index.tolist()
    counts = ga_counts.values

    # Color bars by GA
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(ga_sorted)))

    bars = ax2.bar(
        [str(int(ga)) for ga in ga_sorted],
        counts,
        color=colors,
        edgecolor="black",
        alpha=0.8,
    )

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax2.annotate(
            f"{count}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    ax2.set_xlabel("Gestational Age (weeks, floored)", fontsize=12)
    ax2.set_ylabel("Number of Subjects", fontsize=12)
    ax2.set_title("GA Distribution", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    # Add summary statistics (unique subject/sessions)
    total_n = len(ga_unique)
    mean_ga = quality_df.groupby(["subject_id", "session_id"])["GA"].first().mean()
    std_ga = quality_df.groupby(["subject_id", "session_id"])["GA"].first().std()

    ax2.text(
        0.95,
        0.95,
        f"N = {total_n}\nMean = {mean_ga:.1f}\nSD = {std_ga:.1f}",
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )

    plt.suptitle(
        f"Gestational Age Analysis (N={total_n} subjects)",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    return fig


# Now only needs quality_df
fig = plot_ga_distribution(quality_df)
plt.show()


# In[5]:


# SCATTER PLOT: Volume Difference vs GA (Absolute & Relative, Side by Side)
def plot_abs_and_rel_diff_vs_ga(
    split_pair_diffs, tissue="total", labels=True, show_regression=True
):
    """
    Side-by-side scatter plots: absolute and relative volume difference vs GA.
    """
    from sklearn.linear_model import RANSACRegressor, LinearRegression

    tissue_names = {
        "total": "Total Brain",
        "sp": "Subplate",
        "cp": "Cortical Plate",
        "inner": "Inner",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = {"S1-S2": "#e74c3c", "S3-S4": "#3498db"}

    configs = [
        (axes[0], f"abs_diff_{tissue}", "cm³", 1000, "Absolute"),
        (axes[1], f"rel_diff_{tissue}", "%", 1, "Relative"),
    ]

    for ax, col, unit, scale, diff_type in configs:
        # Collect points per split pair for regression
        regression_data = {"S1-S2": {"ga": [], "y": []}, "S3-S4": {"ga": [], "y": []}}

        for (subj, sess), group in split_pair_diffs.groupby(
            ["subject_id", "session_id"]
        ):
            if len(group) < 2:
                continue

            ga = group["GA"].iloc[0]
            s12 = group[group["split_pair"] == "S1-S2"]
            s34 = group[group["split_pair"] == "S3-S4"]

            if len(s12) == 0 or len(s34) == 0:
                continue

            y12 = s12[col].values[0] / scale
            y34 = s34[col].values[0] / scale

            regression_data["S1-S2"]["ga"].append(ga)
            regression_data["S1-S2"]["y"].append(y12)
            regression_data["S3-S4"]["ga"].append(ga)
            regression_data["S3-S4"]["y"].append(y34)

            # Connect with gray line
            ax.plot(
                [ga, ga], [y12, y34], color="gray", alpha=0.4, linewidth=1, zorder=1
            )

            # Plot points
            ax.scatter(
                ga,
                y12,
                c=colors["S1-S2"],
                s=80,
                alpha=0.8,
                edgecolors="black",
                linewidths=0.5,
                zorder=2,
            )
            ax.scatter(
                ga,
                y34,
                c=colors["S3-S4"],
                s=80,
                alpha=0.8,
                edgecolors="black",
                linewidths=0.5,
                zorder=2,
            )

            if labels:
                y_mid = (y12 + y34) / 2
                ax.text(
                    ga + 0.15, y_mid, str(subj)[:6], fontsize=7, alpha=0.6, va="center"
                )

        # RANSAC regression per split pair
        if show_regression:
            for pair_name in ["S1-S2", "S3-S4"]:
                ga_arr = np.array(regression_data[pair_name]["ga"])
                y_arr = np.array(regression_data[pair_name]["y"])

                if len(ga_arr) < 4:
                    continue

                X = ga_arr.reshape(-1, 1)

                ransac = RANSACRegressor(
                    residual_threshold=np.std(y_arr),
                    min_samples=0.7,
                    random_state=42,
                )
                ransac.fit(X, y_arr)

                x_line = np.linspace(ga_arr.min(), ga_arr.max(), 100).reshape(-1, 1)
                y_line = ransac.predict(x_line)

                ax.plot(
                    x_line,
                    y_line,
                    color=colors[pair_name],
                    linestyle="--",
                    linewidth=2,
                    alpha=0.7,
                    zorder=4,
                )

                inlier_mask = ransac.inlier_mask_
                n_in = np.sum(inlier_mask)
                n_out = np.sum(~inlier_mask)

                # Highlight outliers with hollow markers
                if n_out > 0:
                    ax.scatter(
                        ga_arr[~inlier_mask],
                        y_arr[~inlier_mask],
                        s=80,
                        facecolors="none",
                        edgecolors=colors[pair_name],
                        linewidths=2,
                        zorder=5,
                    )

                # Annotate counts
                y_pos = y_line[-1]
                ax.annotate(
                    f"{pair_name}: {n_in}in/{n_out}out",
                    xy=(x_line[-1, 0], y_pos),
                    fontsize=7,
                    alpha=0.8,
                    color=colors[pair_name],
                    va="bottom",
                    xytext=(5, 3),
                    textcoords="offset points",
                )

        # Legend
        handles = []
        handles.append(
            ax.scatter(
                [], [], c=colors["S1-S2"], s=80, label="S1-S2", edgecolors="black"
            )
        )
        handles.append(
            ax.scatter(
                [], [], c=colors["S3-S4"], s=80, label="S3-S4", edgecolors="black"
            )
        )
        if show_regression:
            from matplotlib.lines import Line2D

            handles.append(
                Line2D(
                    [0],
                    [0],
                    color="gray",
                    linestyle="--",
                    linewidth=2,
                    label="RANSAC fit",
                )
            )
        ax.legend(title="Split Pair", loc="upper left", fontsize=10)

        ax.set_xlabel("Gestational Age (weeks)", fontsize=12)
        ax.set_ylabel(f"{diff_type} Volume Difference ({unit})", fontsize=12)
        ax.set_title(f"{diff_type} Difference", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"{tissue_names[tissue]}: Volume Difference vs GA\n(Lines connect same subject)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    return fig


# --- Run ---
fig = plot_abs_and_rel_diff_vs_ga(split_pair_diffs, tissue="total")
plt.show()

fig = plot_abs_and_rel_diff_vs_ga(split_pair_diffs, tissue="sp")
plt.show()

fig = plot_abs_and_rel_diff_vs_ga(split_pair_diffs, tissue="cp")
plt.show()


# In[6]:


# 2a: Dice per model
def plot_dice_by_model(df):
    """Box plot comparing Dice scores across models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = [
        "SP Model - Subplate",
        "SP Model - Cortical Plate",
        "5-Label Model - Cortical Plate",
    ]
    model_labels = [
        "SP Model\nSubplate",
        "SP Model\nCortical Plate",
        "5-Label Model\nCortical Plate",
    ]
    colors = ["#2ecc71", "#3498db", "#9b59b6"]

    data = [df[df["model"] == m]["dice"].values for m in models]

    bp = ax.boxplot(data, tick_labels=model_labels, patch_artist=True, widths=0.6)

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, d in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.4, s=20, color="black", zorder=3)

    means = [np.mean(d) for d in data]
    ax.scatter(
        range(1, len(models) + 1),
        means,
        color="red",
        s=100,
        zorder=5,
        marker="D",
        label="Mean",
    )

    ax.set_ylabel("Dice Coefficient", fontsize=12)
    ax.set_title(
        "Segmentation Reliability: Dice Score by Model", fontsize=14, fontweight="bold"
    )
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.5, label="Threshold (0.8)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")

    for i, (m, d) in enumerate(zip(means, data)):
        ax.annotate(
            f"μ={m:.3f}\nσ={np.std(d):.3f}",
            xy=(i + 1, 0.05),
            ha="center",
            fontsize=9,
            color="darkblue",
        )

    plt.tight_layout()
    return fig


fig = plot_dice_by_model(reliability_df)
plt.show()


# In[7]:


# 2a: relative diff per model (TODO: Change this to absolute difference CHECK NOTES)
def plot_relative_diff_by_model(df):
    """Box plot comparing relative volume difference across models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = ['SP Model - Subplate', 'SP Model - Cortical Plate', '5-Label Model - Cortical Plate']
    model_labels = ['SP Model\nSubplate', 'SP Model\nCortical Plate', '5-Label Model\nCortical Plate']
    colors = ['#2ecc71', '#3498db', '#9b59b6']

    data = [df[df['model'] == m]['relative_diff'].values for m in models]

    bp = ax.boxplot(data, tick_labels=model_labels, patch_artist=True, widths=0.6)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, d in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.4, s=20, color='black', zorder=3)

    means = [np.mean(d) for d in data]
    ax.scatter(range(1, len(models) + 1), means, color='red', s=100, zorder=5, marker='D', label='Mean')

    ax.set_ylabel('Relative Volume Difference (%)', fontsize=12)
    ax.set_title('Segmentation Reliability: Volume Consistency by Model', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig

fig = plot_relative_diff_by_model(reliability_df)
plt.show()


# In[8]:


# 2a: Abs native volume diference across splits (subjects connected) TODO: Add subject identifier
def _get_subject_colors(quality_df):
    subjects = get_unique_subjects(quality_df)
    n_subjects = len(subjects)
    subject_cmap = plt.cm.get_cmap("tab20", n_subjects)
    subject_colors = {subj: subject_cmap(i) for i, subj in enumerate(subjects)}
    return subject_colors


def _plot_measures_across_splits(quality_df, measure_configs, subject_colors, title, ylabel, labels = False):
    """
    Internal helper to plot boxplots with connected subject lines across splits.

    Parameters:
    -----------
    quality_df : pd.DataFrame
        Long-format dataframe
    measure_configs : list of tuples
        Format: [(column_name, display_title), ...]
    title : str
        Super title for the figure
    ylabel : str
        Label for the y-axis
    """
    splits = get_available_splits(quality_df)
    subjects = get_unique_subjects(quality_df)

    # Distinct colors for boxes and subjects
    box_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    n_plots = len(measure_configs)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]  # Handle single plot case

    for ax, (col_name, display_name) in zip(axes, measure_configs):
        # 1. Prepare Boxplot Data (Aggregate per split)
        native = str(col_name).startswith("native") #<-Check if the col is a native value
        split_data = []
        for split in splits:
            # Use existing helper to get data for this split
            values = get_column_by_split(quality_df, col_name, split).dropna().values
            if native:
                values = values / 1000 #<- mm^3 to cm^3
            split_data.append(values)

        # 2. Draw Boxplot
        if any(len(d) > 0 for d in split_data):
            bp = ax.boxplot(
                split_data, tick_labels=splits, patch_artist=True, widths=0.6
            )
            for patch, color in zip(bp["boxes"], box_colors[: len(splits)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.4)

        # 3. Draw Connected Lines per Subject
        for subj_id, ses_id in subjects:
            color = subject_colors[(subj_id, ses_id)]
            x_pos, y_val = [], []

            for i, split in enumerate(splits):
                # Use existing helper to get specific row for subject+split
                row = get_split_data(quality_df, subj_id, ses_id, split)

                # Check if row exists and column is valid/not-NaN
                if (
                    row is not None
                    and col_name in row.index
                    and pd.notna(row[col_name])
                ):
                    x_pos.append(i + 1)
                    if native:
                        y_val.append(row[col_name] / 1000)  # <- mm^3 to cm^3
                    else:
                        y_val.append(row[col_name]) 

            if x_pos:
                # Scatter points
                ax.scatter(
                    x_pos,
                    y_val,
                    color=color,
                    s=40,
                    alpha=0.8,
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=3,
                )
                # Connected lines - only connect S1-S2 and S3-S4
                # Split into two groups: [S1, S2] and [S3, S4]
                for start_idx in [0, 2]:  # S1 at index 0, S3 at index 2
                    group_x = []
                    group_y = []
                    for j in range(len(x_pos)):
                        if x_pos[j] in [start_idx + 1, start_idx + 2]:  # S1-S2 or S3-S4
                            group_x.append(x_pos[j])
                            group_y.append(y_val[j])
                    if len(group_x) >= 2:
                        ax.plot(group_x, group_y, color=color, alpha=0.5, linewidth=1, zorder=2)

                # Add subject_id label to the right of the line
                if labels:
                    if len(x_pos) > 0:
                        ax.text(x_pos[-1] + 0.1, y_val[-1], f'{subj_id}', 
                            fontsize=8, va='center', ha='left', color=color, alpha=0.7)

        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel("Split", fontsize=12)
        ax.set_title(f"{display_name} by Split", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_voxel_counts_by_split(quality_df, colors):
    """Plot raw voxel counts per split for SP and CP."""
    return _plot_measures_across_splits(
        quality_df,
        measure_configs=[
            ("sp_volume_voxels", "Subplate"),
            ("cp_volume_voxels", "Cortical Plate"),
            ("inner_volume_voxels", "Inner"),
        ],
        subject_colors=colors,
        title="Raw Voxel Counts Across Splits",
        ylabel="Voxel Count",
        labels=True
    )


def plot_native_volume_by_split(quality_df, colors):
    """Plot native volumes (mm³) per split for all tissues."""
    return _plot_measures_across_splits(
        quality_df,
        measure_configs=[
            ("native_vol_sp", "Subplate"),
            ("native_vol_cp", "Cortical Plate"),
            ("native_vol_inner", "Inner"),
        ],
        subject_colors=colors,
        title="Native Volumes Across Splits",
        ylabel="Native Volume (cm³)",
        labels=True
    )

colors = _get_subject_colors(quality_df)

fig1 = plot_voxel_counts_by_split(quality_df, colors)
plt.show()

fig2 = plot_native_volume_by_split(quality_df, colors)
plt.show()


# In[9]:


# VOLUME BAR PLOT COMPARISON PER SPLIT (GROUPED)
def plot_per_subject_voxel_counts(quality_df, tissue="sp"):
    """
    Bar plot showing voxel counts per subject for all 4 splits.
    Allows visual comparison of volume consistency within each subject.
    Subjects are ordered by mean voxel count (ascending).

    Parameters:
    -----------
    quality_df : pd.DataFrame
        Long-format dataframe
    tissue : str
        'sp' for subplate, 'cp' for cortical plate, 'inner' for inner tissue
    """
    tissue_names = {"sp": "Subplate", "cp": "Cortical Plate", "inner": "Inner"}
    col_name = f"{tissue}_volume_voxels"
    tissue_name = tissue_names.get(tissue, tissue)

    subjects = get_unique_subjects(quality_df)
    splits = get_available_splits(quality_df)
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    # Calculate mean voxel count per subject for ordering
    subject_means = {}
    for (subj_id, ses_id) in subjects:
        vals = []
        for split in splits:
            row = get_split_data(quality_df, subj_id, ses_id, split)
            if row is not None and col_name in row.index and pd.notna(row[col_name]):
                vals.append(row[col_name])
        subject_means[(subj_id, ses_id)] = np.mean(vals) if vals else 0

    # Sort subjects by mean voxel count (ascending)
    subjects_sorted = sorted(subjects, key=lambda s: subject_means[s])

    fig, ax = plt.subplots(figsize=(16, 8))

    x = np.arange(len(subjects_sorted))
    width = 0.2

    for i, split in enumerate(splits):
        values = []
        for (subj_id, ses_id) in subjects_sorted:
            row = get_split_data(quality_df, subj_id, ses_id, split)
            if row is not None and col_name in row.index:
                values.append(row[col_name])
            else:
                values.append(np.nan)

        offset = (i - 1.5) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=split,
            color=colors[i],
            alpha=0.8,
            edgecolor="black",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(s) for s, _ in subjects_sorted], rotation=45, ha="right", fontsize=9
    )
    ax.set_xlabel("Subject ID", fontsize=12)
    ax.set_ylabel("Voxel Count", fontsize=12)
    ax.set_title(
        f"{tissue_name} Voxel Count per Subject Across Splits",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(title="Split", loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig

def plot_per_subject_native_volume(quality_df, tissue="sp"):
    """
    Bar plot showing native volumes per subject for all splits.
    Allows visual comparison of volume consistency within each subject.
    Subjects are ordered by mean native volume (ascending).

    Parameters:
    -----------
    quality_df : pd.DataFrame
        Long-format dataframe
    tissue : str
        'sp' for subplate, 'cp' for cortical plate, 'inner' for inner tissue
    """
    tissue_names = {"sp": "Subplate", "cp": "Cortical Plate", "inner": "Inner"}
    col_name = f"native_vol_{tissue}"
    tissue_name = tissue_names.get(tissue, tissue)

    subjects = get_unique_subjects(quality_df)
    splits = get_available_splits(quality_df)
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    # Calculate mean native volume per subject for ordering
    subject_means = {}
    for (subj_id, ses_id) in subjects:
        vals = []
        for split in splits:
            row = get_split_data(quality_df, subj_id, ses_id, split)
            if row is not None and col_name in row.index and pd.notna(row[col_name]):
                vals.append(row[col_name])
        subject_means[(subj_id, ses_id)] = np.mean(vals) if vals else 0

    # Sort subjects by mean native volume (ascending)
    subjects_sorted = sorted(subjects, key=lambda s: subject_means[s])

    fig, ax = plt.subplots(figsize=(16, 8))

    x = np.arange(len(subjects_sorted))
    width = 0.2

    for i, split in enumerate(splits):
        values = []
        for (subj_id, ses_id) in subjects_sorted:
            row = get_split_data(quality_df, subj_id, ses_id, split)
            if row is not None and col_name in row.index:
                # Convert mm³ to cm³
                values.append(row[col_name] / 1000)
            else:
                values.append(np.nan)

        offset = (i - 1.5) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=split,
            color=colors[i],
            alpha=0.8,
            edgecolor="black",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(s) for s, _ in subjects_sorted], rotation=45, ha="right", fontsize=9
    )
    ax.set_xlabel("Subject ID", fontsize=12)
    ax.set_ylabel("Native Volume (cm³)", fontsize=12)
    ax.set_title(
        f"{tissue_name} Native Volume per Subject Across Splits",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(title="Split", loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


## NATIVE VOLUMES
print("=" * 60)
print(" NATIVE VOLUMES")
print("=" * 60)

# Plot for Subplate
fig = plot_per_subject_native_volume(quality_df, tissue="sp")
plt.show()

# Plot for Cortical Plate
fig = plot_per_subject_native_volume(quality_df, tissue="cp")
plt.show()

# Plot for Inner tissue
fig = plot_per_subject_native_volume(quality_df, tissue="inner")
plt.show()

# Voxel counts
print("=" * 60)
print(" VOXEL COUNTS ")
print("=" * 60)

# Plot for Subplate
fig = plot_per_subject_voxel_counts(quality_df, tissue="sp")
plt.show()

# Plot for Cortical Plate
fig = plot_per_subject_voxel_counts(quality_df, tissue="cp")
plt.show()

# Plot for Inner tissue
fig = plot_per_subject_voxel_counts(quality_df, tissue="inner")
plt.show()


# In[10]:


# SNR PLOT
def plot_snr_by_split(quality_df):
    """
    Box plot showing SNR per split for each tissue type.
    Now includes all three tissues: Subplate, Cortical Plate, and Inner.

    Uses long-format quality_df with columns: snr_subplate, snr_cortical_plate, snr_inner
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    splits = get_available_splits(quality_df)
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    tissue_configs = [
        ("snr_subplate", "Subplate SNR", axes[0]),
        ("snr_cortical_plate", "Cortical Plate SNR", axes[1]),
    ]

    for col_name, title, ax in tissue_configs:
        # Collect data for box plots
        split_data = []
        for split in splits:
            values = get_column_by_split(quality_df, col_name, split).dropna().values
            split_data.append(values)

        if any(len(d) > 0 for d in split_data):
            bp = ax.boxplot(
                split_data, tick_labels=splits, patch_artist=True, widths=0.6
            )
            for patch, color in zip(bp["boxes"], colors[: len(splits)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Add scatter points
            for i, d in enumerate(split_data):
                if len(d) > 0:
                    x = np.random.normal(i + 1, 0.04, size=len(d))
                    ax.scatter(x, d, alpha=0.5, s=30, color="black", zorder=3)

        ax.set_ylabel("Signal-to-Noise Ratio", fontsize=12)
        ax.set_xlabel("Split", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "SNR Comparison Across Splits (All Tissues)",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    return fig


fig = plot_snr_by_split(quality_df)
plt.show()


# In[11]:


# SNR VS NATIVE VOLUME COMPARISON (ABSOLUTE DIFFERENCE)
from sklearn.linear_model import RANSACRegressor, LinearRegression


def plot_snr_volume_correlation(quality_df, split_pair=("S1", "S2")):
    """
    Scatter plots showing correlation between SNR difference and absolute native volume difference.
    Uses RANSAC robust linear regression.
    """
    split1, split2 = split_pair
    pair_name = f"{split1}-{split2}"

    tissues = [
        ("snr_subplate", "native_vol_sp", "Subplate"),
        ("snr_cortical_plate", "native_vol_cp", "Cortical Plate"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (snr_col, vol_col, tissue_name) in zip(axes, tissues):
        # Compute SNR difference (absolute)
        snr_diff_df = _compute_split_pair_diff(
            quality_df, snr_col, split1, split2, relative=False
        )
        snr_diff_df = snr_diff_df.rename(columns={"diff": "snr_diff"})

        # Compute native volume absolute difference (in mm³, will convert to cm³)
        vol_diff_df = _compute_split_pair_diff(
            quality_df, vol_col, split1, split2, relative=False
        )
        vol_diff_df = vol_diff_df.rename(columns={"diff": "vol_absdiff"})

        # Merge on subject_id AND session_id
        merged = snr_diff_df[["subject_id", "session_id", "label", "snr_diff"]].merge(
            vol_diff_df[["subject_id", "session_id", "vol_absdiff"]],
            on=["subject_id", "session_id"],
        )

        if len(merged) < 3:
            ax.text(
                0.5,
                0.5,
                f"Insufficient data\n(N={len(merged)})",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title(f"{tissue_name}: {pair_name}")
            continue

        x = merged["snr_diff"].values
        y = merged["vol_absdiff"].values / 1000  # Convert mm³ to cm³

        ax.scatter(x, y, s=80, alpha=0.7, edgecolors="black")

        # Add subject labels
        for _, row in merged.iterrows():
            ax.annotate(
                row["label"][:8],
                (row["snr_diff"], row["vol_absdiff"] / 1000),
                fontsize=7,
                alpha=0.7,
                xytext=(3, 3),
                textcoords="offset points",
            )

        # Pearson correlation (on original data)
        r, p = stats.pearsonr(x, y)

        # RANSAC robust linear regression
        X_reshaped = x.reshape(-1, 1)

        ransac = RANSACRegressor(
            residual_threshold= np.std(y) * 1.5,
            min_samples=0.5,  # Use at least 50% of data for fitting
            random_state=42,
        )
        ransac.fit(X_reshaped, y)

        # Get RANSAC statistics
        inlier_mask = ransac.inlier_mask_
        n_inliers = np.sum(inlier_mask)
        r2_score = ransac.score(X_reshaped, y)

        # Plot RANSAC regression line
        x_line = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
        y_line = ransac.predict(x_line)
        ax.plot(x_line, y_line, "r--", alpha=0.8, linewidth=2, label="RANSAC fit")

        # # SIMPLE LINEAR REGRESSION <=========================================
        # z = np.polyfit(x, y, 1)
        # p_line = np.poly1d(z)
        # ax.plot(
        #     x_line.flatten(),
        #     p_line(x_line.flatten()),
        #     "b-",
        #     alpha=0.6,
        #     linewidth=2,
        #     label="OLS fit",
        # )

        # Mark outliers differently
        outlier_mask = ~inlier_mask
        if np.any(outlier_mask):
            ax.scatter(
                x[outlier_mask],
                y[outlier_mask],
                s=80,
                alpha=0.7,
                edgecolors="red",
                linewidths=2,
                facecolors="none",
                label=f"Outliers ({np.sum(outlier_mask)})",
            )

        ax.set_xlabel(f"|SNR Difference| ({pair_name})", fontsize=11)
        ax.set_ylabel(f"Native Volume Abs. Diff (cm³) ({pair_name})", fontsize=11)
        ax.set_title(
            f"{tissue_name}\nPearson: r={r:.3f}, p={p:.3f} | RANSAC: R²={r2_score:.3f}, inliers={n_inliers}/{len(x)}",
            fontsize=11,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    plt.suptitle(
        f"Image Quality (SNR) vs Segmentation Reliability ({pair_name})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    return fig


# Plot for S1-S2 (Independent Pair 1)
fig = plot_snr_volume_correlation(quality_df, split_pair=("S1", "S2"))
plt.show()

# Plot for S3-S4 (Independent Pair 2)
fig = plot_snr_volume_correlation(quality_df, split_pair=("S3", "S4"))
plt.show()


# In[12]:


# === CELL 9: Correlation Matrix Heatmaps (S1-S2 and S3-S4) ===
from statsmodels.stats.multitest import multipletests


def plot_correlation_matrices(quality_df, infodump_df, print_diagnostics=True):
    """
    Create two correlation matrix heatmaps - one for S1-S2 and one for S3-S4.

    Each shows relationships between:
    - Native volume absolute difference (SP and CP) in cm³
    - GA (Gestational Age)
    - QA difference (between splits)
    - QA mean (overall reconstruction quality)
    - SNR difference

    Uses long-format quality_df and computes differences on the fly.
    P-values are corrected for multiple comparisons using FDR (Benjamini-Hochberg).

    Diagonal cells show mean ± SD for each variable instead of r=1.00.
    """
    # Prepare infodump data
    infodump = infodump_df.copy()
    infodump["subject_id"] = infodump["subject_id"].astype(str)
    infodump["session_id"] = infodump["session_id"].astype(str)

    # Calculate QA differences
    if "QA_S1" in infodump.columns and "QA_S2" in infodump.columns:
        infodump["qa_diff_S1S2"] = abs(infodump["QA_S1"] - infodump["QA_S2"])
    if "QA_S3" in infodump.columns and "QA_S4" in infodump.columns:
        infodump["qa_diff_S3S4"] = abs(infodump["QA_S3"] - infodump["QA_S4"])

    # QA means should already be in the data as QA12_mean and QA34_mean
    if "QA12_mean" not in infodump.columns and "QA_S1" in infodump.columns:
        infodump["QA12_mean"] = (infodump["QA_S1"] + infodump["QA_S2"]) / 2
    if "QA34_mean" not in infodump.columns and "QA_S3" in infodump.columns:
        infodump["QA34_mean"] = (infodump["QA_S3"] + infodump["QA_S4"]) / 2

    # Build analysis dataframes for each split pair
    split_configs = {
        "S1-S2": {
            "splits": ("S1", "S2"),
            "qa_diff_col": "qa_diff_S1S2",
            "qa_mean_col": "QA12_mean",
        },
        "S3-S4": {
            "splits": ("S3", "S4"),
            "qa_diff_col": "qa_diff_S3S4",
            "qa_mean_col": "QA34_mean",
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax_idx, (pair_name, config) in enumerate(split_configs.items()):
        ax = axes[ax_idx]
        split1, split2 = config["splits"]

        if print_diagnostics:
            print(f"\n{'='*60}")
            print(f"Processing {pair_name}")
            print(f"{'='*60}")

        # Compute native volume ABSOLUTE differences (then convert to cm³)
        vol_sp_diff = _compute_split_pair_diff(
            quality_df, "native_vol_sp", split1, split2, relative=False
        )
        vol_sp_diff["vol_sp_absdiff"] = vol_sp_diff["diff"] / 1000  # mm³ to cm³
        vol_sp_diff["subject_id"] = vol_sp_diff["subject_id"].astype(str)
        vol_sp_diff["session_id"] = vol_sp_diff["session_id"].astype(str)

        vol_cp_diff = _compute_split_pair_diff(
            quality_df, "native_vol_cp", split1, split2, relative=False
        )
        vol_cp_diff["vol_cp_absdiff"] = vol_cp_diff["diff"] / 1000  # mm³ to cm³
        vol_cp_diff["subject_id"] = vol_cp_diff["subject_id"].astype(str)
        vol_cp_diff["session_id"] = vol_cp_diff["session_id"].astype(str)

        # Compute SNR differences (absolute)
        snr_diff = _compute_split_pair_diff(
            quality_df, "snr_subplate", split1, split2, relative=False
        )
        snr_diff = snr_diff.rename(columns={"diff": "snr_diff"})
        snr_diff["subject_id"] = snr_diff["subject_id"].astype(str)
        snr_diff["session_id"] = snr_diff["session_id"].astype(str)

        if print_diagnostics:
            print(f"\nVolume SP diff subjects: {len(vol_sp_diff)}")
            print(f"Volume CP diff subjects: {len(vol_cp_diff)}")
            print(f"SNR diff subjects: {len(snr_diff)}")

        # Merge all data on BOTH subject_id AND session_id
        merged = (
            vol_sp_diff[["subject_id", "session_id", "vol_sp_absdiff"]]
            .merge(
                vol_cp_diff[["subject_id", "session_id", "vol_cp_absdiff"]],
                on=["subject_id", "session_id"],
                how="inner",
            )
            .merge(
                snr_diff[["subject_id", "session_id", "snr_diff"]],
                on=["subject_id", "session_id"],
                how="inner",
            )
        )

        if print_diagnostics:
            print(f"After merging volume and SNR diffs: {len(merged)} subjects")

        # Add infodump columns - merge on BOTH subject_id AND session_id
        cols_to_add = ["subject_id", "session_id", "GA"]
        if config["qa_diff_col"] in infodump.columns:
            cols_to_add.append(config["qa_diff_col"])
        if config["qa_mean_col"] in infodump.columns:
            cols_to_add.append(config["qa_mean_col"])

        merged = merged.merge(
            infodump[cols_to_add].drop_duplicates(),
            on=["subject_id", "session_id"],
            how="left",
        )

        if print_diagnostics:
            print(f"After merging with infodump: {len(merged)} subjects")

        # Define columns and labels for correlation matrix
        columns = ["vol_sp_absdiff", "vol_cp_absdiff", "snr_diff"]
        labels = ["Vol Diff\nSP (cm³)", "Vol Diff\nCP (cm³)", "SNR Diff"]

        if config["qa_diff_col"] in merged.columns:
            columns.append(config["qa_diff_col"])
            labels.append("QA Diff")

        if config["qa_mean_col"] in merged.columns:
            columns.append(config["qa_mean_col"])
            labels.append("QA Mean")

        if "GA" in merged.columns:
            columns.append("GA")
            labels.append("GA")

        # Filter to available columns
        available_cols = [c for c in columns if c in merged.columns]
        available_labels = [
            labels[i] for i, c in enumerate(columns) if c in merged.columns
        ]

        if len(available_cols) < 2:
            ax.text(
                0.5,
                0.5,
                f"Insufficient data for {pair_name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title(f"{pair_name} Correlation Matrix")
            continue

        # Get complete cases
        corr_data = merged[available_cols].dropna()

        if print_diagnostics:
            print(f"Complete cases for correlation: {len(corr_data)}")
            print(f"\n--- Diagnostic Table for {pair_name} ---")
            display_df = merged[["subject_id", "session_id"] + available_cols].copy()
            display_df = display_df.round(3)
            print(display_df.to_string(index=False))

            # Print descriptive statistics
            print(f"\n--- Descriptive Statistics for {pair_name} ---")
            desc_stats = corr_data.describe().T[["mean", "std", "min", "max"]]
            desc_stats.index = available_labels
            print(desc_stats.round(3).to_string())
            print()

        if len(corr_data) < 3:
            ax.text(
                0.5,
                0.5,
                f"Not enough complete cases\n(N={len(corr_data)})",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title(f"{pair_name} Correlation Matrix")
            continue

        # Compute correlation matrix
        corr_matrix = corr_data.corr(method="pearson")

        # Compute mean and std for each variable (for diagonal display)
        var_means = corr_data.mean()
        var_stds = corr_data.std()

        # Compute p-values and collect for FDR correction
        n_vars = len(available_cols)
        p_matrix = np.zeros((n_vars, n_vars))
        p_values_list = []  # For FDR correction
        p_indices = []  # Track which cells each p-value belongs to

        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    p_matrix[i, j] = 1.0  # Diagonal
                elif i < j:  # Only compute upper triangle
                    _, p = stats.pearsonr(
                        corr_data[available_cols[i]], corr_data[available_cols[j]]
                    )
                    p_matrix[i, j] = p
                    p_matrix[j, i] = p  # Symmetric
                    p_values_list.append(p)
                    p_indices.append((i, j))

        # Apply FDR correction (Benjamini-Hochberg)
        if len(p_values_list) > 0:
            _, p_corrected, _, _ = multipletests(p_values_list, method="fdr_bh")

            # Create corrected p-value matrix
            p_corrected_matrix = np.ones((n_vars, n_vars))
            for idx, (i, j) in enumerate(p_indices):
                p_corrected_matrix[i, j] = p_corrected[idx]
                p_corrected_matrix[j, i] = p_corrected[idx]
        else:
            p_corrected_matrix = p_matrix.copy()

        if print_diagnostics:
            print(f"--- Raw p-values (upper triangle) ---")
            for idx, (i, j) in enumerate(p_indices):
                print(
                    f"  {available_labels[i]} vs {available_labels[j]}: p={p_values_list[idx]:.4f} -> FDR corrected: {p_corrected[idx]:.4f}"
                )
            print()

        # Plot heatmap
        im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

        # Set ticks
        ax.set_xticks(range(len(available_labels)))
        ax.set_yticks(range(len(available_labels)))
        ax.set_xticklabels(available_labels, fontsize=10)
        ax.set_yticklabels(available_labels, fontsize=10)

        # Add correlation values and significance stars (using FDR-corrected p-values)
        # Diagonal cells show mean ± SD instead of r=1.00
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    # Diagonal: show mean ± SD
                    mean_val = var_means[available_cols[i]]
                    std_val = var_stds[available_cols[i]]

                    # Format based on magnitude (avoid overly long numbers)
                    if abs(mean_val) >= 100:
                        text = f"μ={mean_val:.0f}\nσ={std_val:.0f}"
                    elif abs(mean_val) >= 10:
                        text = f"μ={mean_val:.1f}\nσ={std_val:.1f}"
                    else:
                        text = f"μ={mean_val:.2f}\nσ={std_val:.2f}"

                    # Diagonal is white (r=1), so use black text
                    ax.text(
                        j,
                        i,
                        text,
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=9,
                        fontweight="bold",
                    )
                else:
                    # Off-diagonal: show correlation and significance
                    r = corr_matrix.iloc[i, j]
                    p_corr = p_corrected_matrix[i, j]

                    # Significance stars based on FDR-corrected p-values
                    if p_corr < 0.001:
                        sig = "***"
                    elif p_corr < 0.01:
                        sig = "**"
                    elif p_corr < 0.05:
                        sig = "*"
                    else:
                        sig = ""

                    text_color = "white" if abs(r) > 0.5 else "black"
                    ax.text(
                        j,
                        i,
                        f"{r:.2f}\n{sig}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=11,
                        fontweight="bold",
                    )

        ax.set_title(
            f"{pair_name} (N={len(corr_data)})", fontsize=14, fontweight="bold"
        )

    # Add colorbar manually to avoid tight_layout issues
    cbar = fig.colorbar(
        im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04, shrink=0.8
    )
    cbar.set_label("Pearson Correlation (r)", fontsize=11)

    plt.suptitle(
        "Correlation Matrices: Independent Split Pairs (Absolute Volume Diff)\n"
        "Diagonal: μ=mean, σ=SD | Off-diagonal: * p<0.05, ** p<0.01, *** p<0.001 (FDR corrected)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    # Use constrained_layout or manual adjustment instead of tight_layout
    fig.subplots_adjust(left=0.08, right=0.85, top=0.88, bottom=0.08, wspace=0.3)

    return fig


fig = plot_correlation_matrices(quality_df, infodump_df, print_diagnostics=True)
plt.show()


# # CNR calculation
# Applies dilation to mask to obtain outside voxels and calculates constrast difference

# In[ ]:


# CNR vs GA Scatter Plot
def plot_cnr_vs_ga(quality_df):
    """
    Scatter plot showing CNR (Contrast-to-Noise Ratio) vs Gestational Age.
    Shows SP/IZ CNR values across all splits with subject-level grouping.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Get unique subjects for color mapping
    unique_subjects = quality_df[["subject_id", "session_id"]].drop_duplicates()
    n_subjects = len(unique_subjects)
    subject_cmap = plt.cm.get_cmap("tab20", n_subjects)

    # Create color dictionary
    subject_colors = {}
    for i, (_, row) in enumerate(unique_subjects.iterrows()):
        subject_colors[(row["subject_id"], row["session_id"])] = subject_cmap(i)

    # Group by subject and session
    grouped = quality_df.groupby(["subject_id", "session_id"])

    for (subj_id, sess_id), group in grouped:
        color = subject_colors[(subj_id, sess_id)]
        ga_values = group["GA"].values
        cnr_values = group["sp_iz_cnr"].values

        # Plot individual points (4 splits per subject)
        ax.scatter(
            ga_values,
            cnr_values,
            c=[color],
            s=60,
            alpha=0.5,
            edgecolors="black",
            linewidths=0.5,
        )

        # Plot mean as larger marker
        mean_ga = ga_values.mean()
        mean_cnr = cnr_values.mean()
        ax.scatter(
            mean_ga,
            mean_cnr,
            c=[color],
            s=150,
            alpha=0.9,
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
        )

    # Calculate and display correlation using subject means
    subject_means = (
        quality_df.groupby(["subject_id", "session_id"])
        .agg({"GA": "mean", "sp_iz_cnr": "mean"})
        .dropna()
    )

    if len(subject_means) >= 3:
        r, p = stats.pearsonr(subject_means["GA"], subject_means["sp_iz_cnr"])

        # Add regression line
        z = np.polyfit(subject_means["GA"], subject_means["sp_iz_cnr"], 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(subject_means["GA"].min(), subject_means["GA"].max(), 100)
        ax.plot(
            x_line, p_line(x_line), "r--", alpha=0.8, linewidth=2, label="Linear fit"
        )

        # Add correlation text box
        ax.text(
            0.05,
            0.95,
            f"r = {r:.3f}\np = {p:.3f}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    ax.set_xlabel("Gestational Age (weeks)", fontsize=12)
    ax.set_ylabel("CNR (Subplate/Inner Zone)", fontsize=12)
    ax.set_title(
        "Contrast-to-Noise Ratio vs Gestational Age\n(Large dots = subject mean, small dots = individual splits)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    return fig


# Generate plot
fig = plot_cnr_vs_ga(quality_df)
plt.show()


# In[10]:


# CNR Bar Plot per Subject (Organized by GA)
def plot_cnr_per_subject_by_ga(quality_df):
  """
  Bar plot showing CNR (SP/IZ) per subject across all 4 splits.
  Subjects are ordered by GA (ascending).
  """
  subjects = get_unique_subjects(quality_df)
  splits = get_available_splits(quality_df)
  colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

  # Calculate mean GA per subject for ordering
  subject_gas = {}
  for (subj_id, ses_id) in subjects:
    rows = quality_df[
      (quality_df["subject_id"] == subj_id)
      & (quality_df["session_id"] == ses_id)
    ]
    if len(rows) > 0:
      subject_gas[(subj_id, ses_id)] = rows["GA"].iloc[0]

  # Sort subjects by GA (ascending)
  subjects_sorted = sorted(subjects, key=lambda s: subject_gas.get(s, 0))

  fig, ax = plt.subplots(figsize=(16, 8))

  x = np.arange(len(subjects_sorted))
  width = 0.2

  for i, split in enumerate(splits):
    values = []
    for (subj_id, ses_id) in subjects_sorted:
      row = get_split_data(quality_df, subj_id, ses_id, split)
      if row is not None and "sp_iz_cnr" in row.index:
        values.append(row["sp_iz_cnr"])
      else:
        values.append(np.nan)

    offset = (i - 1.5) * width
    ax.bar(
      x + offset,
      values,
      width,
      label=split,
      color=colors[i],
      alpha=0.8,
      edgecolor="black",
    )

  # Create x-tick labels with subject_id and GA
  tick_labels = [
    f"{subj}\nGA={subject_gas.get((subj, ses), 0):.1f}"
    for subj, ses in subjects_sorted
  ]

  ax.set_xticks(x)
  ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
  ax.set_xlabel("SubjectID", fontsize=12)
  ax.set_ylabel("CNR (Subplate/Inner Zone)", fontsize=12)
  ax.set_title(
    "CNR per Subject Across Splits (Ordered by GA)",
    fontsize=14,
    fontweight="bold",
  )
  ax.legend(title="Split", loc="upper left")
  ax.grid(True, alpha=0.3, axis="y")

  plt.tight_layout()
  return fig


fig = plot_cnr_per_subject_by_ga(quality_df)
plt.show()


# # MULTIVARIATE REGRESSION ANALYSIS

# In[12]:


# Imports
from tabulate import tabulate 
from IPython.display import display


# ### Model: Vol_diff ~ GA + SNR_diff + QA_diff + stack_count

# In[ ]:


# Multivariate Analysis - Volume Difference Predictors

# Model: Vol_diff ~ GA + SNR_diff + QA_diff + stack_count
#
# Run 4 models:
#   1. SP, S1-S2
#   2. SP, S3-S4
#   3. CP, S1-S2
#   4. CP, S3-S4
# =============================================================================


import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load split comparison data (has volume differences already computed)
split_comp_df = pd.read_csv("../data/split_comparision_data.csv")


def prepare_multivariate_data(quality_df, infodump_df, split_comp_df, split_pair):
    """
    Prepare merged dataset for multivariate regression.

    Parameters:
    -----------
    quality_df : pd.DataFrame
        Image quality metrics (SNR per split)
    infodump_df : pd.DataFrame
        Raw subject data (stack_count, QA values)
    split_comp_df : pd.DataFrame
        Split comparison data (volume differences)
    split_pair : str
        'S1-S2' or 'S3-S4'

    Returns:
    --------
    pd.DataFrame with columns: vol_diff_sp, vol_diff_cp, GA, snr_diff, qa_diff, stack_count
    """
    # Filter split comparison data for this pair
    comp_data = split_comp_df[split_comp_df["split_pair"] == split_pair].copy()
    comp_data["subject_id"] = comp_data["subject_id"].astype(str)
    comp_data["session_id"] = comp_data["session_id"].astype(str)

    # Get the two splits
    if split_pair == "S1-S2":
        split1, split2 = "S1", "S2"
        qa_diff_col = "qa_diff_S1S2"
        qa1_col, qa2_col = "QA_S1", "QA_S2"
    else:
        split1, split2 = "S3", "S4"
        qa_diff_col = "qa_diff_S3S4"
        qa1_col, qa2_col = "QA_S3", "QA_S4"

    # Compute SNR differences from quality_df
    quality_df = quality_df.copy()
    quality_df["subject_id"] = quality_df["subject_id"].astype(str)
    quality_df["session_id"] = quality_df["session_id"].astype(str)

    # Get SNR for each split
    snr_s1 = quality_df[quality_df["split"] == split1][
        ["subject_id", "session_id", "snr_subplate", "snr_cortical_plate"]
    ].copy()
    snr_s1 = snr_s1.rename(
        columns={"snr_subplate": "snr_sp_1", "snr_cortical_plate": "snr_cp_1"}
    )

    snr_s2 = quality_df[quality_df["split"] == split2][
        ["subject_id", "session_id", "snr_subplate", "snr_cortical_plate"]
    ].copy()
    snr_s2 = snr_s2.rename(
        columns={"snr_subplate": "snr_sp_2", "snr_cortical_plate": "snr_cp_2"}
    )

    # Merge SNR data
    snr_merged = snr_s1.merge(snr_s2, on=["subject_id", "session_id"], how="inner")
    snr_merged["snr_diff_sp"] = abs(snr_merged["snr_sp_1"] - snr_merged["snr_sp_2"])
    snr_merged["snr_diff_cp"] = abs(snr_merged["snr_cp_1"] - snr_merged["snr_cp_2"])

    # Prepare infodump data
    infodump = infodump_df.copy()
    infodump["subject_id"] = infodump["subject_id"].astype(str)
    infodump["session_id"] = infodump["session_id"].astype(str)

    # Compute QA difference
    infodump[qa_diff_col] = abs(infodump[qa1_col] - infodump[qa2_col])

    # Merge everything
    merged = comp_data.merge(
        snr_merged[["subject_id", "session_id", "snr_diff_sp", "snr_diff_cp"]],
        on=["subject_id", "session_id"],
        how="inner",
    )

    merged = merged.merge(
        infodump[["subject_id", "session_id", "stack_count", qa_diff_col]],
        on=["subject_id", "session_id"],
        how="inner",
    )

    # Rename for clarity
    merged = merged.rename(
        columns={
            "abs_diff_sp": "vol_diff_sp",
            "abs_diff_cp": "vol_diff_cp",
            qa_diff_col: "qa_diff",
        }
    )

    # Convert volume from mm³ to cm³
    merged["vol_diff_sp"] = merged["vol_diff_sp"] / 1000
    merged["vol_diff_cp"] = merged["vol_diff_cp"] / 1000

    return merged


def compute_vif(X):
    """Compute Variance Inflation Factor for each predictor."""
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
    ]
    return vif_data


def run_multivariate_model(df, tissue, split_pair, print_results=True):
    """
    Run OLS regression: vol_diff ~ GA + snr_diff + qa_diff + stack_count
    Returns a dictionary with results and prints a publication-style table.
    """
    # Select appropriate columns
    vol_col = f"vol_diff_{tissue}"
    snr_col = f"snr_diff_{tissue}"

    # Create analysis dataframe
    analysis_df = df[["GA", snr_col, "qa_diff", "stack_count", vol_col]].dropna().copy()
    analysis_df = analysis_df.rename(columns={snr_col: "snr_diff", vol_col: "vol_diff"})

    n = len(analysis_df)

    if n < 10:
        print(f"WARNING: Only {n} complete cases for {tissue.upper()} {split_pair}")
        return None

    # Fit full model
    formula = "vol_diff ~ GA + snr_diff + qa_diff + stack_count"
    model = smf.ols(formula, data=analysis_df).fit()

    # Compute VIF for collinearity check
    X = analysis_df[["GA", "snr_diff", "qa_diff", "stack_count"]]
    vif_df = compute_vif(X)

    if print_results:

        def format_pval(p):
            if p < 0.001:
                return "<0.001 ***"
            if p < 0.01:
                return f"{p:.3f} **"
            if p < 0.05:
                return f"{p:.3f} *"
            return f"{p:.3f}"

        print(f"\n{'='*60}")
        print(f"Table: Regression of {tissue.upper()} Volume Diff ({split_pair})")
        print(f"{'='*60}")

        table_data = []
        for var in model.params.index:
            table_data.append(
                [
                    var,
                    f"{model.params[var]:.3f}",
                    f"{model.bse[var]:.3f}",
                    f"{model.tvalues[var]:.2f}",
                    format_pval(model.pvalues[var]),
                ]
            )

        print(
            tabulate(
                table_data,
                headers=["Variable", "Coef", "SE", "t", "p-value"],
                floatfmt=".3f",
            )
        )

        # Model Fit Statistics Footer
        print(f"{'R-squared':<15} {model.rsquared:>10.3f}")
        print(f"{'Adj. R-squared':<15} {model.rsquared_adj:>10.3f}")
        print(f"{'N':<15} {n:>10}")
        print(f"{'F-statistic':<15} {model.fvalue:>10.2f}")

        print("\nCollinearity Statistics (VIF):")

        # Create a header for the list
        print(f"  {'Variable':<15} {'VIF':>8}")
        print("  " + "-" * 25)

        # Iterate through ALL variables
        for _, row in vif_df.iterrows():
            vif_val = row["VIF"]
            var_name = row["Variable"]

            # Add an asterisk (*) if VIF is high
            highlight = " *" if vif_val > 5 else ""

            print(f"  {var_name:<15} {vif_val:>8.2f}{highlight}")

    return {
        "tissue": tissue,
        "split_pair": split_pair,
        "n": n,
        "model": model,
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "f_stat": model.fvalue,
        "f_pvalue": model.f_pvalue,
        "coefficients": model.params.to_dict(),
        "pvalues": model.pvalues.to_dict(),
        "vif": vif_df,
    }


def compare_models(results_list, variables=None):
    """
    Create comparison table across all 4 models.
    
    Parameters:
    -----------
    results_list : list
        List of result dictionaries from model runs
    variables : list, optional
        List of variable names to compare. If None, automatically extracts
        from the first valid model's coefficients.
    """
    print(f"\n{'='*80}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*80}")

    # Header
    print(f"\n{'Model':<20} {'N':>5} {'R²':>8} {'Adj R²':>8} {'F':>8} {'F p-val':>10}")
    print(f"{'-'*80}")

    for r in results_list:
        if r is not None:
            label = f"{r['tissue'].upper()} ({r['split_pair']})"
            sig = (
                "***"
                if r["f_pvalue"] < 0.001
                else (
                    "**"
                    if r["f_pvalue"] < 0.01
                    else "*" if r["f_pvalue"] < 0.05 else ""
                )
            )
            print(
                f"{label:<20} {r['n']:>5} {r['r_squared']:>8.4f} {r['adj_r_squared']:>8.4f} {r['f_stat']:>8.2f} {r['f_pvalue']:>9.4f}{sig}"
            )

    # Auto-extract variables from first valid model if not provided
    if variables is None:
        for r in results_list:
            if r is not None and "coefficients" in r:
                variables = list(r["coefficients"].keys())
                break
    
    if variables is None:
        print("\nNo valid models to compare coefficients.")
        return

    # Coefficient comparison
    print(f"\n{'-'*80}")
    print("COEFFICIENT COMPARISON (significance: * p<0.05, ** p<0.01, *** p<0.001)")
    print(f"{'-'*80}")

    print(f"\n{'Variable':<20}", end="")
    for r in results_list:
        if r is not None:
            label = f"{r['tissue'].upper()}_{r['split_pair'][:2]}"
            print(f"{label:>15}", end="")
    print()
    print(f"{'-'*80}")

    for var in variables:
        print(f"{var:<20}", end="")
        for r in results_list:
            if r is not None:
                coef = r["coefficients"].get(var, np.nan)
                pval = r["pvalues"].get(var, 1.0)
                if np.isnan(coef):
                    print(f"{'N/A':>15}", end="")
                else:
                    sig = (
                        "***"
                        if pval < 0.001
                        else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                    )
                    print(f"{coef:>12.3f}{sig:>3}", end="")
        print()


def _plot_universal_results(df, tissue, split_pair, model_result, x_axis_col="GA"):
    """
    Universal plotting function for any OLS model.

    Instead of re-predicting (which requires specific column names),
    this extracts the fitted values and residuals directly from the
    statsmodels object and aligns them with the original dataframe
    using the index.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the 'GA' column and the target volume column.
    tissue : str
        'sp' or 'cp'
    split_pair : str
        Label for the title (e.g., 'S1-S2')
    model_result : dict
        The dictionary returned by your runner functions (must contain 'model')
    x_axis_col : str
        The main variable to plot against (default 'GA')
    """
    if model_result is None:
        return None

    model = model_result["model"]
    vol_col = f"vol_diff_{tissue}"

    # 1. Get the indices of the data points actually used in the model
    #    (This automatically accounts for rows dropped due to NaNs in ANY predictor)
    valid_indices = model.fittedvalues.index

    # 2. Subset the original dataframe to match the model data
    #    We only strictly need the x_axis_col and the target vol_col here
    plot_data = df.loc[valid_indices].copy()

    # 3. Add model outputs
    plot_data["predicted"] = model.fittedvalues
    plot_data["residuals"] = model.resid
    plot_data["observed"] = plot_data[vol_col]

    # Visual Setup
    tissue_name = "Subplate" if tissue == "sp" else "Cortical Plate"
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Plot 1: Actual vs Predicted ---
    ax1 = axes[0]
    ax1.scatter(
        plot_data["observed"],
        plot_data["predicted"],
        alpha=0.7,
        s=60,
        edgecolors="black",
    )

    # Perfect prediction line
    min_val = min(plot_data["observed"].min(), plot_data["predicted"].min())
    max_val = max(plot_data["observed"].max(), plot_data["predicted"].max())
    ax1.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect fit")

    ax1.set_xlabel(f"Actual {vol_col} (cm³)", fontsize=11)
    ax1.set_ylabel("Predicted (cm³)", fontsize=11)
    ax1.set_title(
        f"Actual vs Predicted\n$R^2$ = {model.rsquared:.3f}",
        fontsize=12,
        fontweight="bold",
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Volume Diff vs GA (Observed) ---
    ax2 = axes[1]
    ax2.scatter(
        plot_data[x_axis_col],
        plot_data["observed"],
        alpha=0.7,
        s=60,
        edgecolors="black",
        label="Observed Data",
    )

    # Add a simple linear trend line for visualization (Univariate)
    # We use numpy polyfit on the valid data
    if len(plot_data) > 1:
        z = np.polyfit(plot_data[x_axis_col], plot_data["observed"], 1)
        p_line = np.poly1d(z)
        x_range = np.linspace(
            plot_data[x_axis_col].min(), plot_data[x_axis_col].max(), 100
        )
        ax2.plot(
            x_range, p_line(x_range), "r-", linewidth=2, alpha=0.8, label="Linear Trend"
        )

        # Correlation
        r, p = stats.pearsonr(plot_data[x_axis_col], plot_data["observed"])
        corr_label = f"r = {r:.3f}, p = {p:.3f}"
    else:
        corr_label = "Insufficient Data"

    ax2.set_xlabel(f"{x_axis_col} (weeks)", fontsize=11)
    ax2.set_ylabel("Actual Volume Diff (cm³)", fontsize=11)
    ax2.set_title(
        f"Volume Diff vs {x_axis_col}\n{corr_label}", fontsize=12, fontweight="bold"
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Residuals vs GA ---
    ax3 = axes[2]
    ax3.scatter(
        plot_data[x_axis_col],
        plot_data["residuals"],
        alpha=0.7,
        s=60,
        edgecolors="black",
    )
    ax3.axhline(y=0, color="r", linestyle="--", linewidth=2)

    ax3.set_xlabel(f"{x_axis_col} (weeks)", fontsize=11)
    ax3.set_ylabel("Residuals (cm³)", fontsize=11)
    ax3.set_title(f"Residuals vs {x_axis_col}", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3)

    # --- Super Title ---
    # Try to extract the formula from the result dict if it exists, otherwise generic
    formula_text = model_result.get("formula", "Multivariate Model")

    plt.suptitle(
        f"{tissue_name} - {split_pair}\n{formula_text}",
        fontsize=14,
        fontweight="bold",
        y=1.05,
    )
    plt.tight_layout()

    return fig


# =============================================================================
# RUN ANALYSIS
# =============================================================================

print("=" * 80)
print("MULTIVARIATE REGRESSION ANALYSIS")
print("Model: Vol_diff ~ GA + SNR_diff + QA_diff + stack_count")
print("=" * 80)

# Prepare data for each split pair
data_S1S2 = prepare_multivariate_data(quality_df, infodump_df, split_comp_df, "S1-S2")
data_S3S4 = prepare_multivariate_data(quality_df, infodump_df, split_comp_df, "S3-S4")

print(f"\nData prepared:")
print(f"  S1-S2: {len(data_S1S2)} subjects")
print(f"  S3-S4: {len(data_S3S4)} subjects")

# Run all 4 models
results = []

# SP S1-S2
results.append(run_multivariate_model(data_S1S2, "sp", "S1-S2"))

# SP S3-S4
results.append(run_multivariate_model(data_S3S4, "sp", "S3-S4"))

# CP S1-S2
results.append(run_multivariate_model(data_S1S2, "cp", "S1-S2"))

# CP S3-S4
results.append(run_multivariate_model(data_S3S4, "cp", "S3-S4"))

# Comparison table
compare_models(results)

# Generate plots
print(f"\n{'='*80}")
print("GENERATING VISUALIZATIONS")
print(f"{'='*80}")

fig1 = _plot_universal_results(data_S1S2, "sp", "S1-S2", results[0])
plt.show()

fig2 = _plot_universal_results(data_S3S4, "sp", "S3-S4", results[1])
plt.show()

fig3 = _plot_universal_results(data_S1S2, "cp", "S1-S2", results[2])
plt.show()

fig4 = _plot_universal_results(data_S3S4, "cp", "S3-S4", results[3])
plt.show()


# ##  Residualization of  stack_count against GA
# 
# Model: Model: Vol_diff ~ GA + SNR_diff + QA_diff + stack_count_resid

# In[ ]:


def residualize(df, target_col, covariate_col, print_diagnostics=True):
    """
    Residualize target_col against covariate_col using OLS regression.

    Returns the residuals of: target ~ covariate
    i.e., the part of target NOT explained by covariate.

    This is useful for removing collinearity: if A and B are correlated,
    residualizing B against A gives you the unique variance in B that A
    doesn't already capture. You can then use A + B_residualized in a model
    without collinearity issues.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing both columns
    target_col : str
        The variable to residualize (e.g., 'stack_count')
    covariate_col : str
        The variable to regress out (e.g., 'GA')
    print_diagnostics : bool
        Whether to print the regression summary

    Returns:
    --------
    pd.Series
        Residuals (same length as df, NaN where input was NaN)

    Example:
    --------
    >>> df['stack_count_resid'] = residualize(df, 'stack_count', 'GA')
    >>> # Now stack_count_resid is uncorrelated with GA
    """
    # Get complete cases
    mask = df[[target_col, covariate_col]].notna().all(axis=1)
    clean = df.loc[mask, [target_col, covariate_col]].copy()

    if len(clean) < 3:
        print(f"WARNING: Only {len(clean)} complete cases for residualization")
        return pd.Series(np.nan, index=df.index)

    # Fit simple OLS: target ~ covariate
    model = smf.ols(f"{target_col} ~ {covariate_col}", data=clean).fit()

    if print_diagnostics:
        r, p = stats.pearsonr(clean[covariate_col], clean[target_col])
        print(f"Residualizing: {target_col} ~ {covariate_col}")
        print(f"  Original correlation: r = {r:.3f}, p = {p:.4f}")
        print(f"  Regression R² = {model.rsquared:.4f}")
        print(
            f"  Slope = {model.params[covariate_col]:.4f} (p = {model.pvalues[covariate_col]:.4f})"
        )
        print(
            f"  Interpretation: {model.rsquared*100:.1f}% of {target_col} variance is shared with {covariate_col}"
        )
        print(
            f"  Residuals capture the remaining {(1-model.rsquared)*100:.1f}% unique to {target_col}"
        )

    # Create output series with NaN for incomplete cases
    residuals = pd.Series(np.nan, index=df.index)
    residuals.loc[mask] = model.resid.values

    return residuals


# =============================================================================
# MODEL WITH RESIDUALIZED STACK COUNT
# =============================================================================
# Model: Vol_diff ~ GA + SNR_diff + QA_diff + stack_count_resid
#
# stack_count_resid = residuals of (stack_count ~ GA)
# This removes the GA-correlated component from stack_count,
# keeping only the unique acquisition-specific variance.
# =============================================================================


def prepare_residualized_data(quality_df, infodump_df, split_comp_df, split_pair):
    """
    Prepare data with residualized stack_count for multivariate regression.
    Same as prepare_multivariate_data but adds stack_count_resid.
    """
    # Use existing preparation function
    merged = prepare_multivariate_data(
        quality_df, infodump_df, split_comp_df, split_pair
    )

    # Residualize stack_count against GA
    print(f"\n--- Residualizing stack_count for {split_pair} ---")
    merged["stack_count_resid"] = residualize(
        merged, "stack_count", "GA", print_diagnostics=True
    )

    # Verify collinearity is resolved
    r_original, _ = stats.pearsonr(
        merged["GA"].dropna(), merged["stack_count"].dropna()
    )

    mask = merged[["GA", "stack_count_resid"]].notna().all(axis=1)
    if mask.sum() >= 3:
        r_resid, _ = stats.pearsonr(
            merged.loc[mask, "GA"], merged.loc[mask, "stack_count_resid"]
        )
        print(f"  Correlation GA vs stack_count (original):      r = {r_original:.4f}")
        print(f"  Correlation GA vs stack_count_resid (cleaned):  r = {r_resid:.4f}")

    return merged


def run_residualized_model(df, tissue, split_pair, print_results=True):
    """
    Run OLS: vol_diff ~ GA + snr_diff + qa_diff + stack_count_resid
    """
    vol_col = f"vol_diff_{tissue}"
    snr_col = f"snr_diff_{tissue}"

    analysis_df = (
        df[["GA", snr_col, "qa_diff", "stack_count_resid", vol_col]].dropna().copy()
    )
    analysis_df = analysis_df.rename(columns={snr_col: "snr_diff", vol_col: "vol_diff"})

    n = len(analysis_df)
    if n < 10:
        print(f"WARNING: Only {n} complete cases for {tissue.upper()} {split_pair}")
        return None

    formula = "vol_diff ~ GA + snr_diff + qa_diff + stack_count_resid"
    model = smf.ols(formula, data=analysis_df).fit()

    X = analysis_df[["GA", "snr_diff", "qa_diff", "stack_count_resid"]]
    vif_df = compute_vif(X)

    if print_results:
        def format_pval(p):
            if p < 0.001:
                return "<0.001 ***"
            if p < 0.01:
                return f"{p:.3f} **"
            if p < 0.05:
                return f"{p:.3f} *"
            return f"{p:.3f}"

        print(f"\n{'='*60}")
        print(f"Table: Regression of {tissue.upper()} Volume Diff ({split_pair})")
        print(f"{'='*60}")

        table_data = []
        for var in model.params.index:
            table_data.append(
                [
                    var,
                    f"{model.params[var]:.3f}",
                    f"{model.bse[var]:.3f}",
                    f"{model.tvalues[var]:.2f}",
                    format_pval(model.pvalues[var]),
                ]
            )

        print(
            tabulate(
                table_data,
                headers=["Variable", "Coef", "SE", "t", "p-value"],
                floatfmt=".3f",
            )
        )

        # Model Fit Statistics Footer
        print(f"{'R-squared':<15} {model.rsquared:>10.3f}")
        print(f"{'Adj. R-squared':<15} {model.rsquared_adj:>10.3f}")
        print(f"{'N':<15} {n:>10}")
        print(f"{'F-statistic':<15} {model.fvalue:>10.2f}")

        print("\nCollinearity Statistics (VIF):")

        # Create a header for the list
        print(f"  {'Variable':<15} {'VIF':>8}")
        print("  " + "-" * 25)

        # Iterate through ALL variables
        for _, row in vif_df.iterrows():
            vif_val = row["VIF"]
            var_name = row["Variable"]

            # Add an asterisk (*) if VIF is high
            highlight = " *" if vif_val > 5 else ""

            print(f"  {var_name:<15} {vif_val:>8.2f}{highlight}")

    return {
        "tissue": tissue,
        "split_pair": split_pair,
        "n": n,
        "model": model,
        "model_name": "Residualized",
        "formula": "Vol_diff ~ GA + SNR_diff + QA_diff + stack_count_resid",
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "f_stat": model.fvalue,
        "f_pvalue": model.f_pvalue,
        "coefficients": model.params.to_dict(),
        "pvalues": model.pvalues.to_dict(),
        "vif": vif_df,
    }


# =============================================================================
# RUN RESIDUALIZED MODELS
# =============================================================================

print("=" * 80)
print("RESIDUALIZED MODEL: Vol_diff ~ GA + SNR_diff + QA_diff + stack_count_resid")
print("(stack_count_resid = stack_count with GA component removed)")
print("=" * 80)

data_resid_S1S2 = prepare_residualized_data(
    quality_df, infodump_df, split_comp_df, "S1-S2"
)
data_resid_S3S4 = prepare_residualized_data(
    quality_df, infodump_df, split_comp_df, "S3-S4"
)

results_resid = []
results_resid.append(run_residualized_model(data_resid_S1S2, "sp", "S1-S2"))
results_resid.append(run_residualized_model(data_resid_S3S4, "sp", "S3-S4"))
results_resid.append(run_residualized_model(data_resid_S1S2, "cp", "S1-S2"))
results_resid.append(run_residualized_model(data_resid_S3S4, "cp", "S3-S4"))

compare_models(results_resid)

# Plots — using dedicated function that includes stack_count_resid in the dataframe
for data, tissue, pair, result in [
    (data_resid_S1S2, "sp", "S1-S2", results_resid[0]),
    (data_resid_S3S4, "sp", "S3-S4", results_resid[1]),
    (data_resid_S1S2, "cp", "S1-S2", results_resid[2]),
    (data_resid_S3S4, "cp", "S3-S4", results_resid[3]),
]:
    if result is not None:
        fig = _plot_universal_results(data, tissue, pair, result)
        if fig:
            plt.show()


# # Model with QA_mean
# ### Model: Vol_diff ~ GA + SNR_diff + QA_mean + stack_count
# Checking for correlation between poor overall quality vs unreliable segmentation 

# In[70]:


# MODEL WITH QA MEAN (overall quality) INSTEAD OF QA DIFF
# =============================================================================
# Model: Vol_diff ~ GA + SNR_diff + QA_mean + stack_count
#
# Hypothesis: Overall reconstruction quality (not quality difference between
# splits) may predict segmentation volume differences. Poor overall quality
# could lead to less reliable segmentations regardless of split.
#
# QA12_mean is used for S1-S2, QA34_mean for S3-S4.
# =============================================================================


def prepare_qa_mean_data(quality_df, infodump_df, split_comp_df, split_pair):
    """
    Prepare data using QA_mean instead of QA_diff.

    Uses QA12_mean for S1-S2 and QA34_mean for S3-S4 from infodump_df.
    """
    # Filter split comparison data for this pair
    comp_data = split_comp_df[split_comp_df["split_pair"] == split_pair].copy()
    comp_data["subject_id"] = comp_data["subject_id"].astype(str)
    comp_data["session_id"] = comp_data["session_id"].astype(str)

    # Get the two splits and corresponding QA mean column
    if split_pair == "S1-S2":
        split1, split2 = "S1", "S2"
        qa_mean_col = "QA12_mean"
    else:
        split1, split2 = "S3", "S4"
        qa_mean_col = "QA34_mean"

    # Compute SNR differences from quality_df (same as original)
    quality_copy = quality_df.copy()
    quality_copy["subject_id"] = quality_copy["subject_id"].astype(str)
    quality_copy["session_id"] = quality_copy["session_id"].astype(str)

    snr_s1 = quality_copy[quality_copy["split"] == split1][
        ["subject_id", "session_id", "snr_subplate", "snr_cortical_plate"]
    ].copy()
    snr_s1 = snr_s1.rename(
        columns={"snr_subplate": "snr_sp_1", "snr_cortical_plate": "snr_cp_1"}
    )

    snr_s2 = quality_copy[quality_copy["split"] == split2][
        ["subject_id", "session_id", "snr_subplate", "snr_cortical_plate"]
    ].copy()
    snr_s2 = snr_s2.rename(
        columns={"snr_subplate": "snr_sp_2", "snr_cortical_plate": "snr_cp_2"}
    )

    snr_merged = snr_s1.merge(snr_s2, on=["subject_id", "session_id"], how="inner")
    snr_merged["snr_diff_sp"] = abs(snr_merged["snr_sp_1"] - snr_merged["snr_sp_2"])
    snr_merged["snr_diff_cp"] = abs(snr_merged["snr_cp_1"] - snr_merged["snr_cp_2"])

    # Prepare infodump data with QA mean
    infodump = infodump_df.copy()
    infodump["subject_id"] = infodump["subject_id"].astype(str)
    infodump["session_id"] = infodump["session_id"].astype(str)

    # Merge everything
    merged = comp_data.merge(
        snr_merged[["subject_id", "session_id", "snr_diff_sp", "snr_diff_cp"]],
        on=["subject_id", "session_id"],
        how="inner",
    )

    merged = merged.merge(
        infodump[["subject_id", "session_id", "stack_count", qa_mean_col]],
        on=["subject_id", "session_id"],
        how="inner",
    )

    # Rename for clarity
    merged = merged.rename(
        columns={
            "abs_diff_sp": "vol_diff_sp",
            "abs_diff_cp": "vol_diff_cp",
            qa_mean_col: "qa_mean",
        }
    )

    # Convert volume from mm³ to cm³
    merged["vol_diff_sp"] = merged["vol_diff_sp"] / 1000
    merged["vol_diff_cp"] = merged["vol_diff_cp"] / 1000

    return merged


def run_qa_mean_model(df, tissue, split_pair, print_results=True):
    """
    Run OLS: vol_diff ~ GA + snr_diff + qa_mean + stack_count
    """
    vol_col = f"vol_diff_{tissue}"
    snr_col = f"snr_diff_{tissue}"

    analysis_df = df[["GA", snr_col, "qa_mean", "stack_count", vol_col]].dropna().copy()
    analysis_df = analysis_df.rename(columns={snr_col: "snr_diff", vol_col: "vol_diff"})

    n = len(analysis_df)
    if n < 10:
        print(f"WARNING: Only {n} complete cases for {tissue.upper()} {split_pair}")
        return None

    formula = "vol_diff ~ GA + snr_diff + qa_mean + stack_count"
    model = smf.ols(formula, data=analysis_df).fit()

    X = analysis_df[["GA", "snr_diff", "qa_mean", "stack_count"]]
    vif_df = compute_vif(X)

    if print_results:

        def format_pval(p):
            if p < 0.001:
                return "<0.001 ***"
            if p < 0.01:
                return f"{p:.3f} **"
            if p < 0.05:
                return f"{p:.3f} *"
            return f"{p:.3f}"

        print(f"\n{'='*60}")
        print(f"Table: Regression of {tissue.upper()} Volume Diff ({split_pair})")
        print(f"{'='*60}")

        table_data = []
        for var in model.params.index:
            table_data.append(
                [
                    var,
                    f"{model.params[var]:.3f}",
                    f"{model.bse[var]:.3f}",
                    f"{model.tvalues[var]:.2f}",
                    format_pval(model.pvalues[var]),
                ]
            )

        print(
            tabulate(
                table_data,
                headers=["Variable", "Coef", "SE", "t", "p-value"],
                floatfmt=".3f",
            )
        )

        # Model Fit Statistics Footer
        print(f"{'R-squared':<15} {model.rsquared:>10.3f}")
        print(f"{'Adj. R-squared':<15} {model.rsquared_adj:>10.3f}")
        print(f"{'N':<15} {n:>10}")
        print(f"{'F-statistic':<15} {model.fvalue:>10.2f}")

        print("\nCollinearity Statistics (VIF):")

        # Create a header for the list
        print(f"  {'Variable':<15} {'VIF':>8}")
        print("  " + "-" * 25)

        # Iterate through ALL variables
        for _, row in vif_df.iterrows():
            vif_val = row["VIF"]
            var_name = row["Variable"]

            # Add an asterisk (*) if VIF is high
            highlight = " *" if vif_val > 5 else ""

            print(f"  {var_name:<15} {vif_val:>8.2f}{highlight}")

    return {
        "tissue": tissue,
        "split_pair": split_pair,
        "n": n,
        "model": model,
        "model_name": "QA-Mean",
        "formula": "Vol_diff ~ GA + SNR_diff + QA_mean + stack_count",
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "f_stat": model.fvalue,
        "f_pvalue": model.f_pvalue,
        "coefficients": model.params.to_dict(),
        "pvalues": model.pvalues.to_dict(),
        "vif": vif_df,
    }


# =============================================================================
# RUN QA-MEAN MODELS
# =============================================================================

print("=" * 80)
print("QA-MEAN MODEL: Vol_diff ~ GA + SNR_diff + QA_mean + stack_count")
print("(Uses overall reconstruction quality instead of quality difference)")
print("=" * 80)

data_qamean_S1S2 = prepare_qa_mean_data(quality_df, infodump_df, split_comp_df, "S1-S2")
data_qamean_S3S4 = prepare_qa_mean_data(quality_df, infodump_df, split_comp_df, "S3-S4")

print(f"\nData prepared:")
print(f"  S1-S2: {len(data_qamean_S1S2)} subjects")
print(f"  S3-S4: {len(data_qamean_S3S4)} subjects")

results_qamean = []
results_qamean.append(run_qa_mean_model(data_qamean_S1S2, "sp", "S1-S2"))
results_qamean.append(run_qa_mean_model(data_qamean_S3S4, "sp", "S3-S4"))
results_qamean.append(run_qa_mean_model(data_qamean_S1S2, "cp", "S1-S2"))
results_qamean.append(run_qa_mean_model(data_qamean_S3S4, "cp", "S3-S4"))

compare_models(results_qamean)

# Plots
for data, tissue, pair, result in [
    (data_qamean_S1S2, "sp", "S1-S2", results_qamean[0]),
    (data_qamean_S3S4, "sp", "S3-S4", results_qamean[1]),
    (data_qamean_S1S2, "cp", "S1-S2", results_qamean[2]),
    (data_qamean_S3S4, "cp", "S3-S4", results_qamean[3]),
]:
    if result is not None:
        fig = _plot_universal_results(data, tissue, pair, result)
        if fig:
            plt.show()


# # Model with CNR (Contrast-to-Noise Ratio)
# ### Model: Vol_diff ~ GA + CNR_diff + QA_diff + stack_count
# Using CNR difference between splits instead of SNR difference
# CNR data comes from split_comparision_data.csv (cnr_diff, cnr_mean columns)

# In[14]:


# MODEL WITH CNR DIFFERENCE
# =============================================================================
# Model: Vol_diff ~ GA + CNR_diff + QA_diff + stack_count


def prepare_cnr_data(quality_df, infodump_df, split_comp_df, split_pair):
    """
    Prepare data using CNR_diff from split comparison data.

    CNR (Contrast-to-Noise Ratio) measures the contrast between subplate
    and inner zone tissues, which is directly relevant to segmentation.

    Parameters:
    -----------
    quality_df : pd.DataFrame
        Image quality metrics
    infodump_df : pd.DataFrame
        Raw subject data (stack_count, QA values)
    split_comp_df : pd.DataFrame
        Split comparison data (already contains cnr_diff, cnr_mean)
    split_pair : str
        'S1-S2' or 'S3-S4'

    Returns:
    --------
    pd.DataFrame with columns: vol_diff_sp, vol_diff_cp, GA, cnr_diff, cnr_mean, qa_diff, stack_count
    """
    # Filter split comparison data for this pair
    comp_data = split_comp_df[split_comp_df["split_pair"] == split_pair].copy()
    comp_data["subject_id"] = comp_data["subject_id"].astype(str)
    comp_data["session_id"] = comp_data["session_id"].astype(str)

    # Get QA diff column name based on split pair
    if split_pair == "S1-S2":
        qa_diff_col = "qa_diff_S1S2"
        qa1_col, qa2_col = "QA_S1", "QA_S2"
    else:
        qa_diff_col = "qa_diff_S3S4"
        qa1_col, qa2_col = "QA_S3", "QA_S4"

    # Prepare infodump data
    infodump = infodump_df.copy()
    infodump["subject_id"] = infodump["subject_id"].astype(str)
    infodump["session_id"] = infodump["session_id"].astype(str)

    # Compute QA difference
    infodump[qa_diff_col] = abs(infodump[qa1_col] - infodump[qa2_col])

    # Merge with infodump to get stack_count and qa_diff
    merged = comp_data.merge(
        infodump[["subject_id", "session_id", "stack_count", qa_diff_col, "GA"]],
        on=["subject_id", "session_id"],
        how="inner",
    )

    # Use GA from infodump if not in comp_data, otherwise keep comp_data's GA
    if "GA_y" in merged.columns:
        merged["GA"] = merged["GA_y"].fillna(merged["GA_x"])
        merged = merged.drop(columns=["GA_x", "GA_y"])

    # Rename for clarity
    merged = merged.rename(
        columns={
            "abs_diff_sp": "vol_diff_sp",
            "abs_diff_cp": "vol_diff_cp",
            qa_diff_col: "qa_diff",
        }
    )

    # Convert volume from mm³ to cm³
    merged["vol_diff_sp"] = merged["vol_diff_sp"] / 1000
    merged["vol_diff_cp"] = merged["vol_diff_cp"] / 1000

    return merged


def run_cnr_diff_model(df, tissue, split_pair, print_results=True):
    """
    Run OLS: vol_diff ~ GA + cnr_diff + qa_diff + stack_count

    Uses CNR difference between splits as predictor.
    """
    vol_col = f"vol_diff_{tissue}"

    analysis_df = (
        df[["GA", "cnr_diff", "qa_diff", "stack_count", vol_col]].dropna().copy()
    )
    analysis_df = analysis_df.rename(columns={vol_col: "vol_diff"})

    n = len(analysis_df)
    if n < 10:
        print(f"WARNING: Only {n} complete cases for {tissue.upper()} {split_pair}")
        return None

    formula = "vol_diff ~ GA + cnr_diff + qa_diff + stack_count"
    model = smf.ols(formula, data=analysis_df).fit()

    X = analysis_df[["GA", "cnr_diff", "qa_diff", "stack_count"]]
    vif_df = compute_vif(X)

    if print_results:

        def format_pval(p):
            if p < 0.001:
                return "<0.001 ***"
            if p < 0.01:
                return f"{p:.3f} **"
            if p < 0.05:
                return f"{p:.3f} *"
            return f"{p:.3f}"

        print(f"\n{'='*60}")
        print(f"CNR-DIFF MODEL: {tissue.upper()} Volume Diff ({split_pair})")
        print(f"{'='*60}")

        table_data = []
        for var in model.params.index:
            table_data.append(
                [
                    var,
                    f"{model.params[var]:.3f}",
                    f"{model.bse[var]:.3f}",
                    f"{model.tvalues[var]:.2f}",
                    format_pval(model.pvalues[var]),
                ]
            )

        print(
            tabulate(
                table_data,
                headers=["Variable", "Coef", "SE", "t", "p-value"],
                floatfmt=".3f",
            )
        )

        print(f"{'R-squared':<15} {model.rsquared:>10.3f}")
        print(f"{'Adj. R-squared':<15} {model.rsquared_adj:>10.3f}")
        print(f"{'N':<15} {n:>10}")
        print(f"{'F-statistic':<15} {model.fvalue:>10.2f}")

        print("\nCollinearity Statistics (VIF):")
        print(f"  {'Variable':<15} {'VIF':>8}")
        print("  " + "-" * 25)
        for _, row in vif_df.iterrows():
            vif_val = row["VIF"]
            var_name = row["Variable"]
            highlight = " *" if vif_val > 5 else ""
            print(f"  {var_name:<15} {vif_val:>8.2f}{highlight}")

    return {
        "tissue": tissue,
        "split_pair": split_pair,
        "n": n,
        "model": model,
        "model_name": "CNR-Diff",
        "formula": "Vol_diff ~ GA + CNR_diff + QA_diff + stack_count",
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "f_stat": model.fvalue,
        "f_pvalue": model.f_pvalue,
        "coefficients": model.params.to_dict(),
        "pvalues": model.pvalues.to_dict(),
        "vif": vif_df,
    }


def run_cnr_mean_model(df, tissue, split_pair, print_results=True):
    """
    Run OLS: vol_diff ~ GA + cnr_mean + qa_diff + stack_count

    Uses mean CNR across splits as predictor (overall image quality).
    """
    vol_col = f"vol_diff_{tissue}"

    analysis_df = (
        df[["GA", "cnr_mean", "qa_diff", "stack_count", vol_col]].dropna().copy()
    )
    analysis_df = analysis_df.rename(columns={vol_col: "vol_diff"})

    n = len(analysis_df)
    if n < 10:
        print(f"WARNING: Only {n} complete cases for {tissue.upper()} {split_pair}")
        return None

    formula = "vol_diff ~ GA + cnr_mean + qa_diff + stack_count"
    model = smf.ols(formula, data=analysis_df).fit()

    X = analysis_df[["GA", "cnr_mean", "qa_diff", "stack_count"]]
    vif_df = compute_vif(X)

    if print_results:

        def format_pval(p):
            if p < 0.001:
                return "<0.001 ***"
            if p < 0.01:
                return f"{p:.3f} **"
            if p < 0.05:
                return f"{p:.3f} *"
            return f"{p:.3f}"

        print(f"\n{'='*60}")
        print(f"CNR-MEAN MODEL: {tissue.upper()} Volume Diff ({split_pair})")
        print(f"{'='*60}")

        table_data = []
        for var in model.params.index:
            table_data.append(
                [
                    var,
                    f"{model.params[var]:.3f}",
                    f"{model.bse[var]:.3f}",
                    f"{model.tvalues[var]:.2f}",
                    format_pval(model.pvalues[var]),
                ]
            )

        print(
            tabulate(
                table_data,
                headers=["Variable", "Coef", "SE", "t", "p-value"],
                floatfmt=".3f",
            )
        )

        print(f"{'R-squared':<15} {model.rsquared:>10.3f}")
        print(f"{'Adj. R-squared':<15} {model.rsquared_adj:>10.3f}")
        print(f"{'N':<15} {n:>10}")
        print(f"{'F-statistic':<15} {model.fvalue:>10.2f}")

        print("\nCollinearity Statistics (VIF):")
        print(f"  {'Variable':<15} {'VIF':>8}")
        print("  " + "-" * 25)
        for _, row in vif_df.iterrows():
            vif_val = row["VIF"]
            var_name = row["Variable"]
            highlight = " *" if vif_val > 5 else ""
            print(f"  {var_name:<15} {vif_val:>8.2f}{highlight}")

    return {
        "tissue": tissue,
        "split_pair": split_pair,
        "n": n,
        "model": model,
        "model_name": "CNR-Mean",
        "formula": "Vol_diff ~ GA + CNR_mean + QA_diff + stack_count",
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "f_stat": model.fvalue,
        "f_pvalue": model.f_pvalue,
        "coefficients": model.params.to_dict(),
        "pvalues": model.pvalues.to_dict(),
        "vif": vif_df,
    }

print("=" * 80)
print("CNR-DIFF MODEL: Vol_diff ~ GA + CNR_diff + QA_diff + stack_count")
print("(Uses CNR difference between splits - contrast-based quality metric)")
print("=" * 80)

data_cnr_S1S2 = prepare_cnr_data(quality_df, infodump_df, split_comp_df, "S1-S2")
data_cnr_S3S4 = prepare_cnr_data(quality_df, infodump_df, split_comp_df, "S3-S4")

print(f"\nData prepared:")
print(f"  S1-S2: {len(data_cnr_S1S2)} subjects")
print(f"  S3-S4: {len(data_cnr_S3S4)} subjects")

# Run CNR-Diff models (4 combinations)
results_cnr_diff = []
results_cnr_diff.append(run_cnr_diff_model(data_cnr_S1S2, "sp", "S1-S2"))
results_cnr_diff.append(run_cnr_diff_model(data_cnr_S3S4, "sp", "S3-S4"))
results_cnr_diff.append(run_cnr_diff_model(data_cnr_S1S2, "cp", "S1-S2"))
results_cnr_diff.append(run_cnr_diff_model(data_cnr_S3S4, "cp", "S3-S4"))

compare_models(results_cnr_diff)

# Plots for CNR-Diff
for data, tissue, pair, result in [
    (data_cnr_S1S2, "sp", "S1-S2", results_cnr_diff[0]),
    (data_cnr_S3S4, "sp", "S3-S4", results_cnr_diff[1]),
    (data_cnr_S1S2, "cp", "S1-S2", results_cnr_diff[2]),
    (data_cnr_S3S4, "cp", "S3-S4", results_cnr_diff[3]),
]:
    if result is not None:
        fig = _plot_universal_results(data, tissue, pair, result)
        if fig:
            plt.show()


# ## Model with CNR mean
# 

# In[15]:


print("=" * 80)
print("CNR-MEAN MODEL: Vol_diff ~ GA + mean + QA_diff + stack_count")
print("(Uses CNR difference between splits - contrast-based quality metric)")
print("=" * 80)

data_cnr_S1S2 = prepare_cnr_data(quality_df, infodump_df, split_comp_df, "S1-S2")
data_cnr_S3S4 = prepare_cnr_data(quality_df, infodump_df, split_comp_df, "S3-S4")

print(f"\nData prepared:")
print(f"  S1-S2: {len(data_cnr_S1S2)} subjects")
print(f"  S3-S4: {len(data_cnr_S3S4)} subjects")

# Run CNR-Diff models (4 combinations)
results_cnr_mean = []
results_cnr_mean.append(run_cnr_mean_model(data_cnr_S1S2, "sp", "S1-S2"))
results_cnr_mean.append(run_cnr_mean_model(data_cnr_S3S4, "sp", "S3-S4"))
results_cnr_mean.append(run_cnr_mean_model(data_cnr_S1S2, "cp", "S1-S2"))
results_cnr_mean.append(run_cnr_mean_model(data_cnr_S3S4, "cp", "S3-S4"))

compare_models(results_cnr_mean)

# Plots for CNR-Diff
for data, tissue, pair, result in [
    (data_cnr_S1S2, "sp", "S1-S2", results_cnr_mean[0]),
    (data_cnr_S3S4, "sp", "S3-S4", results_cnr_mean[1]),
    (data_cnr_S1S2, "cp", "S1-S2", results_cnr_mean[2]),
    (data_cnr_S3S4, "cp", "S3-S4", results_cnr_mean[3]),
]:
    if result is not None:
        fig = _plot_universal_results(data, tissue, pair, result)
        if fig:
            plt.show()


# ### Results

# In[18]:


def print_final_summary(reliability_df, quality_df, infodump_df):
    """Print a final summary of key findings."""
    print("\n" + "=" * 80)
    print("FINAL SUMMARY REPORT")
    print("=" * 80)

    n_subjects = reliability_df["subject_id"].nunique()
    n_quality = len(quality_df)

    print(
        f"\nDataset: {n_subjects} subjects with reliability metrics, {n_quality} with quality metrics"
    )

    # GA range
    if "GA" in infodump_df.columns:
        ga_min = infodump_df["GA"].min()
        ga_max = infodump_df["GA"].max()
        print(f"Gestational Age range: {ga_min:.1f} - {ga_max:.1f} weeks")

    print("\n--- RELIABILITY RANKING (by mean Dice) ---")
    model_means = (
        reliability_df.groupby("model")["dice"].mean().sort_values(ascending=False)
    )
    for i, (model, mean_dice) in enumerate(model_means.items(), 1):
        std = reliability_df[reliability_df["model"] == model]["dice"].std()
        print(f"  {i}. {model}: {mean_dice:.4f} ± {std:.4f}")

    print("\n--- RELIABILITY RANKING (by mean Voxel Rel. Diff - lower is better) ---")
    model_vox = (
        reliability_df.groupby("model")["relative_diff"]
        .mean()
        .sort_values(ascending=True)
    )
    for i, (model, mean_vox) in enumerate(model_vox.items(), 1):
        std = reliability_df[reliability_df["model"] == model]["relative_diff"].std()
        print(f"  {i}. {model}: {mean_vox:.2f}% ± {std:.2f}%")

    # SNR summary
    if "snr_sp_mean" in quality_df.columns:
        print(f"\n--- SNR SUMMARY ---")
        print(
            f"  Mean SNR (Subplate): {quality_df['snr_sp_mean'].mean():.2f} ± {quality_df['snr_sp_mean'].std():.2f}"
        )
        print(
            f"  Mean SNR (Cortical Plate): {quality_df['snr_cp_mean'].mean():.2f} ± {quality_df['snr_cp_mean'].std():.2f}"
        )

    print("\n" + "=" * 80)


print_final_summary(reliability_df, quality_df, infodump_df)

