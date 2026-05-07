"""Slide-ready helpers for regression presentation."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def regression_to_table(results, model_name: str = "") -> pd.DataFrame:
    """Convert a fitted statsmodels OLS result into a slide-ready DataFrame."""
    ci = results.conf_int()
    table = pd.DataFrame(
        {
            "Coef": results.params.round(3),
            "SE": results.bse.round(3),
            "t": results.tvalues.round(2),
            "p": results.pvalues.round(3),
            "CI_low": ci[0].round(2),
            "CI_high": ci[1].round(2),
        }
    )
    table["sig"] = table["p"].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    )
    table.attrs["model_name"] = model_name
    table.attrs["R2"] = round(results.rsquared, 3)
    table.attrs["F_p"] = round(results.f_pvalue, 4)
    table.attrs["N"] = int(results.nobs)
    return table


def coef_forest(results, title: str, ax=None, drop_intercept: bool = True):
    """Forest plot of regression coefficients with 95% CIs. Returns the axis."""
    coefs = results.params.copy()
    ci = results.conf_int()
    if drop_intercept:
        for c in ("const", "Intercept"):
            if c in coefs.index:
                coefs = coefs.drop(c)
                ci = ci.drop(c)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.5 * len(coefs) + 1.2))

    err_low = (coefs - ci[0]).values
    err_high = (ci[1] - coefs).values
    ax.errorbar(
        coefs.values,
        range(len(coefs)),
        xerr=[err_low, err_high],
        fmt="o",
        color="black",
        capsize=4,
        markersize=6,
    )
    ax.axvline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_yticks(range(len(coefs)))
    ax.set_yticklabels(coefs.index)
    ax.set_xlabel("Coefficient (95% CI)")
    ax.set_title(title)
    ax.invert_yaxis()
    return ax


def fit_and_present(
    outcome: str,
    predictors: list,
    df_s12: pd.DataFrame,
    df_s34: pd.DataFrame,
    model_label: str,
    save_prefix: str,
    assets_dir: Path,
):
    """Fit OLS for both split pairs, save tables + side-by-side forest plot.

    Returns a dict of {"S1-S2": (results, table), "S3-S4": (results, table)}.
    """
    import statsmodels.api as sm
    from IPython.display import display

    output = {}
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 0.5 * len(predictors) + 2), sharex=False
    )

    for ax, df, pair in zip(axes, [df_s12, df_s34], ["S1-S2", "S3-S4"]):
        cols = [outcome] + predictors
        sub = df[cols].dropna()
        X = sm.add_constant(sub[predictors])
        y = sub[outcome]
        res = sm.OLS(y, X).fit()
        table = regression_to_table(res, f"{model_label} | {pair}")

        coef_forest(
            res,
            f"{pair}: R²={res.rsquared:.2f}, p={res.f_pvalue:.3f}, N={int(res.nobs)}",
            ax=ax,
        )

        table_path = assets_dir / f"{save_prefix}_{pair.replace('-', '')}_table.csv"
        table.to_csv(table_path)

        output[pair] = (res, table)
        print(f"\n=== {model_label} | {pair} | "
              f"R²={res.rsquared:.3f} | F={res.fvalue:.2f} | p={res.f_pvalue:.4f} | "
              f"N={int(res.nobs)} ===")
        display(table)

    fig.suptitle(
        f"{model_label}\n{outcome} ~ {' + '.join(predictors)}",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    # fig_path = assets_dir / f"{save_prefix}_forest.png"
    # fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.show()
    # print(f"  → saved forest plot: {fig_path.name}")

    return output


def diagnostics_grid(
    results_by_pair: dict,
    df_s12: pd.DataFrame,
    df_s34: pd.DataFrame,
    outcome: str,
    predictors: list,
    model_label: str,
    n_outliers: int = 3,
    id_col: str = "subject_id",
):
    """2x4 diagnostic panel for one model fit on both split pairs.

    Rows: S1-S2, S3-S4. Columns: Predicted vs Actual, Residuals vs Fitted,
    QQ-plot of residuals, Residuals vs GA. Top `n_outliers` points by
    absolute studentized residual are highlighted in red and annotated with
    their `id_col` value across all four panels.

    Returns the figure (does not save).
    """
    import numpy as np
    import statsmodels.api as sm
    from scipy import stats

    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    pairs = [("S1-S2", df_s12), ("S3-S4", df_s34)]

    for row_idx, (pair, df) in enumerate(pairs):
        res, _ = results_by_pair[pair]
        keep_cols = [outcome] + predictors + ([id_col] if id_col in df.columns else [])
        sub = df[keep_cols].dropna(subset=[outcome] + predictors).reset_index(drop=True)
        X = sm.add_constant(sub[predictors])
        y = sub[outcome].to_numpy()
        yhat = res.predict(X).to_numpy()
        resid = y - yhat

        # Studentized residuals — used to identify outliers (scale-free)
        infl = res.get_influence()
        student_resid = infl.resid_studentized_internal
        # Indices of the n_outliers largest |studentized residual|
        outlier_idx = np.argsort(np.abs(student_resid))[-n_outliers:][::-1]
        ids = sub[id_col].astype(str).tolist() if id_col in sub.columns else [str(i) for i in range(len(sub))]

        def label_at(ax, x_val, y_val, text):
            ax.annotate(
                text[:10], xy=(x_val, y_val),
                xytext=(6, 6), textcoords="offset points",
                fontsize=7, color="darkred", weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="darkred", alpha=0.85, lw=0.5),
            )

        # ---- Predicted vs Actual ----
        ax = axes[row_idx, 0]
        ax.scatter(y, yhat, alpha=0.75, edgecolor="k", linewidth=0.4)
        ax.scatter(y[outlier_idx], yhat[outlier_idx], color="red", s=60, edgecolor="k", linewidth=0.6, zorder=5, label="Top outliers")
        for i in outlier_idx:
            label_at(ax, y[i], yhat[i], ids[i])
        lim = [min(y.min(), yhat.min()), max(y.max(), yhat.max())]
        pad = 0.05 * (lim[1] - lim[0])
        lim = [lim[0] - pad, lim[1] + pad]
        ax.plot(lim, lim, "r--", alpha=0.6, label="y = x")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"Actual {outcome}")
        ax.set_ylabel(f"Predicted {outcome}")
        ax.set_title(f"{pair}: Predicted vs Actual (R²={res.rsquared:.2f})")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        # ---- Residuals vs Fitted ----
        ax = axes[row_idx, 1]
        ax.scatter(yhat, resid, alpha=0.75, edgecolor="k", linewidth=0.4)
        ax.scatter(yhat[outlier_idx], resid[outlier_idx], color="red", s=60, edgecolor="k", linewidth=0.6, zorder=5)
        for i in outlier_idx:
            label_at(ax, yhat[i], resid[i], ids[i])
        ax.axhline(0, color="r", linestyle="--", alpha=0.6)
        ax.set_xlabel("Fitted")
        ax.set_ylabel("Residual")
        ax.set_title(f"{pair}: Residuals vs Fitted")
        ax.grid(alpha=0.3)

        # ---- QQ plot of residuals (manual, so we can annotate) ----
        ax = axes[row_idx, 2]
        n = len(resid)
        sort_order = np.argsort(resid)
        sorted_resid = resid[sort_order]
        # Theoretical normal quantiles using Filliben/Blom-style plotting positions
        plot_pos = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
        theo_q = stats.norm.ppf(plot_pos)
        ax.scatter(theo_q, sorted_resid, alpha=0.75, edgecolor="k", linewidth=0.4)
        # Reference line through 1st and 3rd quartiles of residuals (R-style)
        q1, q3 = np.percentile(resid, [25, 75])
        slope = (q3 - q1) / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
        intercept = q1 - slope * stats.norm.ppf(0.25)
        x_line = np.array([theo_q.min(), theo_q.max()])
        ax.plot(x_line, slope * x_line + intercept, "r--", alpha=0.6)
        # Highlight + annotate outliers
        for i in outlier_idx:
            pos = int(np.where(sort_order == i)[0][0])
            ax.scatter(theo_q[pos], sorted_resid[pos], color="red", s=60, edgecolor="k", linewidth=0.6, zorder=5)
            label_at(ax, theo_q[pos], sorted_resid[pos], ids[i])
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Ordered residuals")
        ax.set_title(f"{pair}: QQ-plot (residual normality)")
        ax.grid(alpha=0.3)

        # ---- Residuals vs GA ----
        ax = axes[row_idx, 3]
        if "GA" in sub.columns:
            ga_vals = sub["GA"].to_numpy()
            ax.scatter(ga_vals, resid, alpha=0.75, edgecolor="k", linewidth=0.4)
            ax.scatter(ga_vals[outlier_idx], resid[outlier_idx], color="red", s=60, edgecolor="k", linewidth=0.6, zorder=5)
            for i in outlier_idx:
                label_at(ax, ga_vals[i], resid[i], ids[i])
            ax.axhline(0, color="r", linestyle="--", alpha=0.6)
            ax.set_xlabel("GA (weeks)")
            ax.set_ylabel("Residual")
            ax.set_title(f"{pair}: Residuals vs GA")
            ax.grid(alpha=0.3)
        else:
            ax.set_visible(False)

    fig.suptitle(
        f"Diagnostics — {model_label}\n{outcome} ~ {' + '.join(predictors)}"
        f"\n(top {n_outliers} outliers by |studentized residual| labeled in red)",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    return fig


def model_comparison_table(all_results: dict) -> pd.DataFrame:
    """Build an across-model comparison: R², F-p, GA-coef, etc., per pair.

    `all_results` is {model_label: {pair: (results, table)}}.
    """
    rows = []
    for model_label, by_pair in all_results.items():
        for pair, (res, _) in by_pair.items():
            ga_coef = res.params.get("GA", float("nan"))
            ga_p = res.pvalues.get("GA", float("nan"))
            rows.append(
                {
                    "model": model_label,
                    "pair": pair,
                    "N": int(res.nobs),
                    "R2": round(res.rsquared, 3),
                    "adj_R2": round(res.rsquared_adj, 3),
                    "F_p": round(res.f_pvalue, 4),
                    "GA_coef": round(ga_coef, 3),
                    "GA_p": round(ga_p, 4),
                }
            )
    return pd.DataFrame(rows)
