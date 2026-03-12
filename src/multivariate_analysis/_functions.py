from tabulate import tabulate 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load split comparison data (has volume differences already computed)
split_comp_df = pd.read_csv("../data/split_comparision_data.csv")


def prepare_analysis_data(quality_df, infodump_df, split_comp_df, split_pair, 
                         volume_conversion_factor=1000):
    """
    Universal function to prepare merged dataset for multivariate regression.

    Parameters:
    -----------
    quality_df : pd.DataFrame
        Image quality metrics with columns: subject_id, session_id, split, 
        snr_subplate, snr_cortical_plate
    infodump_df : pd.DataFrame
        Raw subject data with QA values and stack_count
    split_comp_df : pd.DataFrame
        Split comparison data with volume differences
    split_pair : str
        Split pair identifier (e.g., 'S1-S2', 'S3-S4')
    volume_conversion_factor : float
        Factor to convert volume units (default: 1000 for mm³ to cm³)

    Returns:
    --------
    pd.DataFrame with standardized columns for analysis
    """
    # Parse split pair
    split1, split2 = split_pair.split('-')
    
    # Filter and prepare comparison data
    comp_data = split_comp_df[split_comp_df["split_pair"] == split_pair].copy()
    comp_data["subject_id"] = comp_data["subject_id"].astype(str)
    comp_data["session_id"] = comp_data["session_id"].astype(str)

    # Prepare quality data
    quality_df = quality_df.copy()
    quality_df["subject_id"] = quality_df["subject_id"].astype(str)
    quality_df["session_id"] = quality_df["session_id"].astype(str)

    # Get SNR for each split
    snr_cols = ["subject_id", "session_id", "snr_subplate", "snr_cortical_plate"]
    snr_s1 = quality_df[quality_df["split"] == split1][snr_cols].copy()
    snr_s2 = quality_df[quality_df["split"] == split2][snr_cols].copy()
    
    # Rename columns
    snr_s1 = snr_s1.rename(columns={
        "snr_subplate": "snr_sp_1", 
        "snr_cortical_plate": "snr_cp_1"
    })
    snr_s2 = snr_s2.rename(columns={
        "snr_subplate": "snr_sp_2", 
        "snr_cortical_plate": "snr_cp_2"
    })

    # Merge SNR data and compute differences
    snr_merged = snr_s1.merge(snr_s2, on=["subject_id", "session_id"], how="inner")
    snr_merged["snr_diff_sp"] = abs(snr_merged["snr_sp_1"] - snr_merged["snr_sp_2"])
    snr_merged["snr_diff_cp"] = abs(snr_merged["snr_cp_1"] - snr_merged["snr_cp_2"])

    # Prepare infodump data
    infodump = infodump_df.copy()
    infodump["subject_id"] = infodump["subject_id"].astype(str)
    infodump["session_id"] = infodump["session_id"].astype(str)

    # Compute QA difference dynamically
    qa1_col = f"QA_{split1}"
    qa2_col = f"QA_{split2}"
    if qa1_col in infodump.columns and qa2_col in infodump.columns:
        infodump["qa_diff"] = abs(infodump[qa1_col] - infodump[qa2_col])
    else:
        # Fallback to existing qa_diff columns if available
        qa_diff_col = f"qa_diff_{split1}{split2}"
        if qa_diff_col in infodump.columns:
            infodump["qa_diff"] = infodump[qa_diff_col]
        else:
            print(f"Warning: No QA difference found for {split_pair}")
            infodump["qa_diff"] = np.nan

    # Merge all data
    merged = comp_data.merge(
        snr_merged[["subject_id", "session_id", "snr_diff_sp", "snr_diff_cp"]],
        on=["subject_id", "session_id"], how="inner"
    )
    
    merge_cols = ["subject_id", "session_id", "stack_count", "qa_diff"]
    if "GA" in infodump.columns:
        merge_cols.append("GA")
    
    merged = merged.merge(
        infodump[merge_cols], on=["subject_id", "session_id"], how="inner"
    )

    # Standardize volume difference column names
    volume_cols = {"abs_diff_sp": "vol_diff_sp", "abs_diff_cp": "vol_diff_cp"}
    merged = merged.rename(columns=volume_cols)

    # Convert volume units
    for col in ["vol_diff_sp", "vol_diff_cp"]:
        if col in merged.columns:
            merged[col] = merged[col] / volume_conversion_factor

    return merged


def compute_vif(X):
    """Compute Variance Inflation Factor for each predictor."""
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
    ]
    return vif_data


def residualize(df, target_col, covariate_col, print_diagnostics=True):
    """
    Residualize target_col against covariate_col using OLS regression.

    Returns the residuals of: target ~ covariate
    i.e., the part of target NOT explained by covariate.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing both columns
    target_col : str
        The variable to residualize
    covariate_col : str
        The variable to regress out
    print_diagnostics : bool
        Whether to print the regression summary

    Returns:
    --------
    pd.Series
        Residuals (same length as df, NaN where input was NaN)
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
        print(f"  Slope = {model.params[covariate_col]:.4f} (p = {model.pvalues[covariate_col]:.4f})")
        print(f"  Residuals capture {(1-model.rsquared)*100:.1f}% unique variance")

    # Create output series with NaN for incomplete cases
    residuals = pd.Series(np.nan, index=df.index)
    residuals.loc[mask] = model.resid.values

    return residuals


def run_regression_model(df, outcome_col, predictors, model_name="", 
                        split_pair="", tissue="", min_cases=10, print_results=True):
    """
    Universal function to run OLS regression with any predictors.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data containing outcome and predictor variables
    outcome_col : str
        Name of the outcome variable column
    predictors : list
        List of predictor variable column names
    model_name : str
        Name/description of the model for output
    split_pair : str
        Split pair identifier for output
    tissue : str
        Tissue type for output
    min_cases : int
        Minimum number of cases required to run model
    print_results : bool
        Whether to print formatted results
        
    Returns:
    --------
    dict : Model results dictionary or None if insufficient data
    """
    # Create analysis dataframe with complete cases
    analysis_cols = [outcome_col] + predictors
    analysis_df = df[analysis_cols].dropna().copy()
    n = len(analysis_df)

    if n < min_cases:
        print(f"WARNING: Only {n} complete cases for {tissue.upper()} {split_pair}")
        return None

    # Build and fit model
    formula = f"{outcome_col} ~ {' + '.join(predictors)}"
    model = smf.ols(formula, data=analysis_df).fit()

    # Compute VIF for collinearity check
    X = analysis_df[predictors]
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

        title = f"Regression of {tissue.upper()} Volume Diff ({split_pair})" if tissue else f"Regression Results ({model_name})"
        print(f"\n{'='*60}")
        print(f"Table: {title}")
        print(f"{'='*60}")

        table_data = []
        for var in model.params.index:
            table_data.append([
                var,
                f"{model.params[var]:.3f}",
                f"{model.bse[var]:.3f}",
                f"{model.tvalues[var]:.2f}",
                format_pval(model.pvalues[var]),
            ])

        print(tabulate(
            table_data,
            headers=["Variable", "Coef", "SE", "t", "p-value"],
            floatfmt=".3f"
        ))

        # Model Fit Statistics
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
        "model_name": model_name,
        "formula": formula,
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


def plot_regression_results(df, outcome_col, model_result, x_axis_col="GA", 
                           tissue_name=None, split_pair=""):
    """
    Universal plotting function for any OLS regression model.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing predictor and outcome variables
    outcome_col : str
        Name of the outcome variable column
    model_result : dict
        The dictionary returned by regression functions (must contain 'model')
    x_axis_col : str
        The main variable to plot against (default 'GA')
    tissue_name : str
        Full tissue name for plot titles (e.g., "Subplate", "Cortical Plate")
    split_pair : str
        Split pair label for title
    """
    if model_result is None:
        return None

    model = model_result["model"]
    if tissue_name is None:
        tissue_name = outcome_col.replace("_", " ").title()

    # 1. Get the indices of the data points actually used in the model
    #    (This automatically accounts for rows dropped due to NaNs in ANY predictor)
    valid_indices = model.fittedvalues.index

    # 2. Subset the original dataframe to match the model data
    plot_data = df.loc[valid_indices].copy()

    # 3. Add model outputs
    plot_data["predicted"] = model.fittedvalues
    plot_data["residuals"] = model.resid
    plot_data["observed"] = plot_data[outcome_col]
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

    ax1.set_xlabel(f"Actual {outcome_col} (cm³)", fontsize=11)
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
    formula_text = model_result.get("formula", "Regression Model")
    title_parts = []
    if tissue_name:
        title_parts.append(tissue_name)
    if split_pair:
        title_parts.append(split_pair)
    
    main_title = " - ".join(title_parts) if title_parts else "Regression Results"
    
    plt.suptitle(
        f"{main_title}\n{formula_text}",
        fontsize=14,
        fontweight="bold",
        y=1.05,
    )
    plt.tight_layout()

    return fig


# ### Model Application: Vol_diff ~ GA + SNR_diff + QA_diff + stack_count