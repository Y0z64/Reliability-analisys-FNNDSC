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
