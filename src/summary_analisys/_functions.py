import pandas as pd

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