from tabulate import tabulate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats

import statsmodels.formula.api as smf
import statsmodels.graphics.regressionplots as smgr
from statsmodels.stats.outliers_influence import variance_inflation_factor


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
    
    # Normalize column names to avoid KeyError from hidden whitespace.
    quality_df = quality_df.copy()
    infodump_df = infodump_df.copy()
    split_comp_df = split_comp_df.copy()
    quality_df.columns = quality_df.columns.astype(str).str.strip()
    infodump_df.columns = infodump_df.columns.astype(str).str.strip()
    split_comp_df.columns = split_comp_df.columns.astype(str).str.strip()

    # Filter and prepare comparison data
    comp_data = split_comp_df[split_comp_df["split_pair"] == split_pair].copy()
    comp_data["subject_id"] = comp_data["subject_id"].astype(str)
    comp_data["session_id"] = comp_data["session_id"].astype(str)

    # Prepare quality data
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

    # Build a robust GA lookup table from available sources.
    ga_sources = []
    if "GA" in comp_data.columns:
        ga_sources.append(comp_data[["subject_id", "session_id", "GA"]].copy())
    if "GA" in infodump.columns:
        ga_sources.append(infodump[["subject_id", "session_id", "GA"]].copy())
    if "GA" in quality_df.columns:
        ga_sources.append(quality_df[["subject_id", "session_id", "GA"]].copy())

    ga_lookup = None
    if ga_sources:
        ga_lookup = pd.concat(ga_sources, ignore_index=True)
        ga_lookup = ga_lookup.dropna(subset=["GA"]).drop_duplicates(
            subset=["subject_id", "session_id"], keep="first"
        )

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

    # Merge all data using subject/session keys, then attach GA consistently.
    merged = comp_data.merge(
        snr_merged[["subject_id", "session_id", "snr_diff_sp", "snr_diff_cp"]],
        on=["subject_id", "session_id"], how="inner"
    )

    infodump_cols = ["subject_id", "session_id", "stack_count", "qa_diff"]
    merged = merged.merge(infodump[infodump_cols], on=["subject_id", "session_id"], how="inner")

    if ga_lookup is not None:
        merged = merged.merge(ga_lookup, on=["subject_id", "session_id"], how="left")

    # Keep a single canonical GA column and cast to numeric when available.
    if "GA" not in merged.columns:
        ga_candidates = [c for c in merged.columns if c.startswith("GA")]
        if ga_candidates:
            merged["GA"] = merged[ga_candidates].bfill(axis=1).iloc[:, 0]
    if "GA" in merged.columns:
        merged["GA"] = pd.to_numeric(merged["GA"], errors="coerce")

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
    
    # Check for missing columns and provide helpful error message
    missing_cols = [col for col in analysis_cols if col not in df.columns]
    if missing_cols:
        
        available_cols = list(df.columns)
        print(f"ERROR: Missing columns: {missing_cols}")
        print(f"Available columns in dataframe: {available_cols}")
        
        # Try to suggest similar column names
        for missing_col in missing_cols:
            suggestions = [col for col in available_cols if missing_col.lower() in col.lower() or col.lower() in missing_col.lower()]
            if suggestions:
                print(f"Possible matches for '{missing_col}': {suggestions}")
        return None
    
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
    Diagnostic plots for an OLS regression model: Actual vs Predicted + Residuals vs GA.
    """
    if model_result is None:
        return None

    model = model_result["model"]
    if tissue_name is None:
        tissue_name = outcome_col.replace("_", " ").title()

    valid_indices = model.fittedvalues.index
    plot_data = df.loc[valid_indices].copy()
    plot_data["predicted"] = model.fittedvalues
    plot_data["residuals"] = model.resid
    plot_data["observed"] = plot_data[outcome_col]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # --- Plot 1: Actual vs Predicted ---
    ax1 = axes[0]
    ax1.scatter(plot_data["observed"], plot_data["predicted"],
                alpha=0.7, s=60, edgecolors="black")
    min_val = min(plot_data["observed"].min(), plot_data["predicted"].min())
    max_val = max(plot_data["observed"].max(), plot_data["predicted"].max())
    ax1.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect fit")
    ax1.set_xlabel(f"Actual {outcome_col} (cm³)", fontsize=11)
    ax1.set_ylabel("Predicted (cm³)", fontsize=11)
    ax1.set_title(f"Actual vs Predicted\n$R^2$ = {model.rsquared:.3f}",
                  fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Residuals vs GA ---
    ax2 = axes[1]
    ax2.scatter(plot_data[x_axis_col], plot_data["residuals"],
                alpha=0.7, s=60, edgecolors="black")
    ax2.axhline(y=0, color="r", linestyle="--", linewidth=2)
    ax2.set_xlabel(f"{x_axis_col} (weeks)", fontsize=11)
    ax2.set_ylabel("Residuals (cm³)", fontsize=11)
    ax2.set_title(f"Residuals vs {x_axis_col}", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    title_parts = []
    if tissue_name:
        title_parts.append(tissue_name)
    if split_pair:
        title_parts.append(split_pair)
    main_title = " - ".join(title_parts) if title_parts else "Regression Results"
    formula_text = model_result.get("formula", "")
    plt.suptitle(f"{main_title}\n{formula_text}", fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    return fig


def plot_coefficients(results_list, title="Coefficient Plot", exclude_intercept=True,
                      predictor_labels=None, save_path=None):
    """
    Forest/coefficient plot for a list of model results.

    Each panel shows only the predictors in that model (no empty rows).
    Points are colored blue (p<0.05) or grey (ns), with 95% CI error bars.
    """
    valid = [r for r in results_list if r is not None]
    if not valid:
        print("No valid models to plot.")
        return None

    n_models = len(valid)
    colors_sig = "#2196F3"
    colors_ns  = "#BDBDBD"

    # Determine per-model predictor height so all panels are the same height
    max_preds = max(
        len([v for v in r["coefficients"] if not (exclude_intercept and v == "Intercept")])
        for r in valid
    )
    fig_h = max(4, max_preds * 0.7 + 1.5)
    fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, fig_h))
    if n_models == 1:
        axes = [axes]

    for ax, r in zip(axes, valid):
        model = r["model"]
        label = f"{r['tissue'].upper()} ({r['split_pair']})"
        r2  = r["r_squared"]
        f_p = r["f_pvalue"]
        if f_p < 0.001:
            p_str = "p<0.001***"
        elif f_p < 0.01:
            p_str = f"p={f_p:.3f}**"
        elif f_p < 0.05:
            p_str = f"p={f_p:.3f}*"
        else:
            p_str = f"p={f_p:.3f}"

        coefs = r["coefficients"]
        pvals = r["pvalues"]
        bse   = model.bse

        # Only this model's own predictors — no empty rows
        model_preds = [v for v in coefs if not (exclude_intercept and v == "Intercept")]
        n_pred = len(model_preds)
        display_names = [(predictor_labels or {}).get(v, v) for v in model_preds]

        ax.axvline(x=0, color="black", linewidth=1, linestyle="--", alpha=0.5)

        for yi, var in enumerate(model_preds):
            c     = coefs[var]
            se    = bse[var]
            p     = pvals[var]
            ci_lo = c - 1.96 * se
            ci_hi = c + 1.96 * se
            color = colors_sig if p < 0.05 else colors_ns
            ax.errorbar(c, yi, xerr=[[c - ci_lo], [ci_hi - c]],
                        fmt="o", color=color, ecolor=color,
                        capsize=4, markersize=7, linewidth=1.8)
            if p < 0.05:
                star = "***" if p < 0.001 else "**" if p < 0.01 else "*"
                x_nudge = (ci_hi - ci_lo) * 0.04 + 0.02
                ax.text(ci_hi + x_nudge, yi, star,
                        va="center", fontsize=9, color=colors_sig)

        ax.set_yticks(range(n_pred))
        ax.set_yticklabels(display_names, fontsize=10)
        ax.set_ylim(-0.5, max_preds - 0.5)   # consistent height across panels
        ax.set_xlabel("Coefficient (cm³)", fontsize=10)
        ax.set_title(f"{label}\n$R^2$={r2:.2f},  {p_str}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25, axis="x")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {save_path}")

    return fig


def plot_r2_comparison(all_results_dict, save_path=None):
    """
    Bar chart comparing R² and Adj R² across all model families and tissue/split combos.

    Parameters
    ----------
    all_results_dict : dict  {model_family_name: results_list}
    save_path : str or None
    """
    records = []
    for family_name, results_list in all_results_dict.items():
        for r in results_list:
            if r is None:
                continue
            records.append({
                "Model": family_name,
                "Label": f"{r['tissue'].upper()} {r['split_pair']}",
                "R²": r["r_squared"],
                "Adj R²": r["adj_r_squared"],
                "F p": r["f_pvalue"],
            })

    df = pd.DataFrame(records)
    labels = df["Label"].unique()
    models = df["Model"].unique()
    n_labels = len(labels)
    n_models = len(models)

    fig, axes = plt.subplots(1, n_labels, figsize=(3.5 * n_labels, 5), sharey=True)
    if n_labels == 1:
        axes = [axes]

    x = np.arange(n_models)
    width = 0.35
    cmap = plt.cm.get_cmap("tab10", n_models)

    for ax, lbl in zip(axes, labels):
        sub = df[df["Label"] == lbl].set_index("Model")
        r2_vals     = [sub.loc[m, "R²"]     if m in sub.index else 0 for m in models]
        adj_r2_vals = [sub.loc[m, "Adj R²"] if m in sub.index else 0 for m in models]
        f_pvals     = [sub.loc[m, "F p"]    if m in sub.index else 1 for m in models]

        bars1 = ax.bar(x - width/2, r2_vals,     width, label="R²",     alpha=0.85)
        bars2 = ax.bar(x + width/2, adj_r2_vals, width, label="Adj R²", alpha=0.6)

        # Significance star on top of R² bar
        for xi, (b, p) in enumerate(zip(bars1, f_pvals)):
            if p < 0.05:
                star = "***" if p < 0.001 else "**" if p < 0.01 else "*"
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                        star, ha="center", va="bottom", fontsize=10, color="#1565C0")

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
        ax.set_title(lbl, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(True, alpha=0.25, axis="y")
        if ax is axes[0]:
            ax.set_ylabel("Explained Variance", fontsize=11)
            ax.legend(fontsize=9)

    fig.suptitle("Model R² Comparison Across Families", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {save_path}")

    return fig


def plot_diagnostic_suite(df, outcome_col, model_result, tissue_name="", split_pair="",
                          predictor_labels=None, save_path_prefix=None):
    """
    Full diagnostic suite for one OLS model. Produces two figures:
      Fig 1 — Partial regression (added-variable) plots, one panel per predictor
      Fig 2 — Residuals vs Fitted | Normal Q-Q | Residual histogram

    Prints Shapiro-Wilk normality test result for capture in notebook output.

    Parameters
    ----------
    df : pd.DataFrame  — must contain all model columns
    outcome_col : str
    model_result : dict from run_regression_model
    tissue_name : str
    split_pair : str
    predictor_labels : dict  — {col_name: display_name}
    save_path_prefix : str or None  — e.g. "../../assets/mv_01_snr_sp_s12"
                        saves as <prefix>_partial.png and <prefix>_resid.png
    """
    if model_result is None:
        return None, None

    model  = model_result["model"]
    label  = f"{tissue_name} ({split_pair})" if tissue_name else split_pair
    fitted = model.fittedvalues
    resid  = model.resid

    predictors = [v for v in model.params.index if v != "Intercept"]
    n_pred = len(predictors)

    # -----------------------------------------------------------------
    # Figure 1: Partial regression plots
    # -----------------------------------------------------------------
    n_cols = min(n_pred, 4)
    n_rows = (n_pred + n_cols - 1) // n_cols
    fig1, axes1 = plt.subplots(n_rows, n_cols,
                                figsize=(4.5 * n_cols, 4.2 * n_rows))
    axes1 = np.array(axes1).flatten()

    # smgr.plot_partregress needs a clean dataframe aligned with the model
    valid_idx  = model.fittedvalues.index
    plot_data  = df.loc[valid_idx].copy()

    for i, pred in enumerate(predictors):
        ax      = axes1[i]
        others  = [v for v in predictors if v != pred]
        smgr.plot_partregress(outcome_col, pred, others,
                              data=plot_data, ax=ax, obs_labels=False)
        dname = (predictor_labels or {}).get(pred, pred)
        ax.set_title(f"Partial: {dname}", fontsize=11, fontweight="bold")
        ax.set_xlabel(f"{dname}  |  other predictors", fontsize=9)
        ax.set_ylabel(f"{outcome_col}  |  other predictors", fontsize=9)
        ax.grid(True, alpha=0.3)

    for i in range(n_pred, len(axes1)):
        axes1[i].set_visible(False)

    fig1.suptitle(f"Partial Regression Plots — {label}",
                  fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path_prefix:
        p = f"{save_path_prefix}_partial.png"
        fig1.savefig(p, bbox_inches="tight", dpi=150)
        print(f"Saved: {p}")

    # -----------------------------------------------------------------
    # Figure 2: Residual diagnostics
    # -----------------------------------------------------------------
    sw_stat, sw_p = stats.shapiro(resid)
    print(f"  Shapiro-Wilk [{label}]: W={sw_stat:.4f}, p={sw_p:.4f}"
          + (" (normal)" if sw_p > 0.05 else " (non-normal)"))

    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.5))

    # Residuals vs Fitted
    ax = axes2[0]
    ax.scatter(fitted, resid, alpha=0.75, s=60, edgecolors="black", color="#546E7A")
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
    # Lowess trend
    from statsmodels.nonparametric.smoothers_lowess import lowess
    sorted_fit = np.sort(fitted)
    lw = lowess(resid.values, fitted.values, frac=0.6)
    ax.plot(lw[:, 0], lw[:, 1], color="orange", linewidth=1.8, label="LOWESS")
    ax.set_xlabel("Fitted Values (cm³)", fontsize=11)
    ax.set_ylabel("Residuals (cm³)", fontsize=11)
    ax.set_title("Residuals vs Fitted", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Normal Q-Q
    ax = axes2[1]
    (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist="norm")
    ax.plot(osm, osr, "o", alpha=0.8, markersize=6, color="#455A64")
    ax.plot(osm, slope * np.array(osm) + intercept, "r--", linewidth=1.5)
    ax.set_xlabel("Theoretical Quantiles", fontsize=11)
    ax.set_ylabel("Sample Quantiles", fontsize=11)
    ax.set_title(f"Normal Q-Q\nShapiro-Wilk  p = {sw_p:.3f}",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Residual histogram
    ax = axes2[2]
    ax.hist(resid, bins=8, edgecolor="black", alpha=0.75, color="#78909C")
    xlo, xhi = ax.get_xlim()
    xs = np.linspace(xlo, xhi, 200)
    mu, sigma = float(resid.mean()), float(resid.std())
    bin_width  = (xhi - xlo) / 8
    ax.plot(xs, stats.norm.pdf(xs, mu, sigma) * len(resid) * bin_width,
            "r-", linewidth=2, label="Normal PDF")
    ax.set_xlabel("Residuals (cm³)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Residual Distribution\nμ={mu:.3f},  σ={sigma:.3f}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig2.suptitle(f"Residual Diagnostics — {label}",
                  fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path_prefix:
        p = f"{save_path_prefix}_resid.png"
        fig2.savefig(p, bbox_inches="tight", dpi=150)
        print(f"Saved: {p}")

    return fig1, fig2


def plot_ols_table(results_list, title="OLS Regression Results", save_path=None):
    """
    Renders the full OLS output as a styled matplotlib table.

    Format (standard econometric style):
      Rows = predictors  — shows  Coef  /  (SE)  /  significance stars
      Footer rows = R², Adj R², N, F-stat, F p-value
      Separate bottom block = VIF per predictor

    Columns = one per model result (tissue × split pair).
    """
    valid = [r for r in results_list if r is not None]
    if not valid:
        return None

    def sig(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        if p < 0.10:  return "†"
        return ""

    # ---- Collect all variable names in order ----
    all_vars = []
    for r in valid:
        for v in r["coefficients"]:
            if v not in all_vars:
                all_vars.append(v)

    col_headers = [f"{r['tissue'].upper()}\n({r['split_pair']})" for r in valid]
    n_cols = len(valid)

    # ---- Build cell data (coef rows: 2 sub-rows each) ----
    # We'll use one row per variable with multi-line text
    coef_rows, coef_labels = [], []
    for var in all_vars:
        row = []
        for r in valid:
            if var in r["coefficients"]:
                c  = r["coefficients"][var]
                se = r["model"].bse[var]
                p  = r["pvalues"][var]
                row.append(f"{c:.3f}{sig(p)}\n({se:.3f})")
            else:
                row.append("—")
        coef_rows.append(row)
        coef_labels.append(var)

    # ---- Fit stats rows ----
    def fmt(v, decimals=3): return f"{v:.{decimals}f}"

    stat_labels = ["", "R²", "Adj R²", "N", "F-stat", "F p-value"]
    stat_rows   = [
        [""] * n_cols,
        [fmt(r["r_squared"])     for r in valid],
        [fmt(r["adj_r_squared"]) for r in valid],
        [str(r["n"])             for r in valid],
        [fmt(r["f_stat"], 2)     for r in valid],
        [f"{r['f_pvalue']:.4f}{sig(r['f_pvalue'])}" for r in valid],
    ]

    # ---- VIF rows (one separator + one row per non-intercept predictor) ----
    vif_labels = [""]          # separator label
    vif_rows   = [[""] * n_cols]  # separator row
    vif_vars   = [v for v in all_vars if v != "Intercept"]
    for var in vif_vars:
        row = []
        for r in valid:
            vif_df = r.get("vif")
            if vif_df is not None and var in vif_df["Variable"].values:
                val = float(vif_df.loc[vif_df["Variable"] == var, "VIF"].values[0])
                flag = " !" if val > 5 else ""
                row.append(f"{val:.1f}{flag}")
            else:
                row.append("—")
        vif_rows.append(row)
        vif_labels.append(f"{var} (VIF)")

    all_rows   = coef_rows   + stat_rows   + vif_rows
    all_labels = coef_labels + stat_labels + vif_labels
    n_rows     = len(all_rows)

    # ---- Render ----
    row_height = 0.55
    fig_h = max(5, n_rows * row_height + 1.5)
    fig, ax = plt.subplots(figsize=(max(8, 3.2 * n_cols + 2), fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=all_rows,
        rowLabels=all_labels,
        colLabels=col_headers,
        cellLoc="center",
        rowLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, row_height / 0.25)

    # Header row styling
    for j in range(n_cols):
        tbl[0, j].set_facecolor("#37474F")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Fit stats block (light blue)
    n_coef = len(coef_rows)
    for i in range(n_coef + 1, n_coef + len(stat_rows) + 1):
        for j in range(-1, n_cols):
            cell = tbl[i, j]
            cell.set_facecolor("#E3F2FD")

    # VIF block (light yellow)
    n_stat = len(stat_rows)
    for i in range(n_coef + n_stat + 1, n_rows + 1):
        for j in range(-1, n_cols):
            tbl[i, j].set_facecolor("#FFFDE7")

    # Highlight significant cells
    for i, row in enumerate(coef_rows):
        for j, cell_val in enumerate(row):
            if any(s in str(cell_val) for s in ["***", "**", "*", "†"]):
                tbl[i + 1, j].set_facecolor("#C8E6C9")

    # Shrink separator rows (first row of each block is empty spacer)
    sep_indices = [n_coef + 1, n_coef + n_stat + 1]
    for si in sep_indices:
        if 0 < si <= n_rows:
            for j in range(-1, n_cols):
                try:
                    tbl[si, j].set_height(tbl[si, j].get_height() * 0.3)
                except KeyError:
                    pass

    note = "† p<0.1   * p<0.05   ** p<0.01   *** p<0.001   |   VIF > 5 flagged with !"
    fig.text(0.5, 0.01, note, ha="center", fontsize=8, style="italic", color="#555")

    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {save_path}")

    return fig


# ### Model Application: Vol_diff ~ GA + SNR_diff + QA_diff + stack_count