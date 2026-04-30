"""
CNR Multivariate Analysis: Full models matching original structure
Vol_diff ~ GA + CNR_diff + QA_diff + stack_count
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import os

# ============================================================================
# DATA LOADING
# ============================================================================

quality_df = pd.read_csv("../../data/image_quality_metrics.csv")
infodump_df = pd.read_csv("../../data/raw_subject_data.csv")
split_pair_diffs = pd.read_csv("../../data/split_comparision_data.csv")

# Merge QA metrics
def compute_qa_diff(row):
    if row['split_pair'] == 'S1-S2':
        return abs(row['QA_S1'] - row['QA_S2'])
    elif row['split_pair'] == 'S3-S4':
        return abs(row['QA_S3'] - row['QA_S4'])
    return None

def compute_qa_mean(row):
    if row['split_pair'] == 'S1-S2':
        return row['QA12_mean']
    elif row['split_pair'] == 'S3-S4':
        return row['QA34_mean']
    return None

qa_cols = ['subject_id', 'session_id', 'QA_S1', 'QA_S2', 'QA_S3', 'QA_S4', 'QA12_mean', 'QA34_mean', 'stack_count']
data = split_pair_diffs.merge(
    infodump_df[qa_cols],
    on=['subject_id', 'session_id'],
    how='left'
)
data['QA_diff'] = data.apply(compute_qa_diff, axis=1)
data['QA_mean'] = data.apply(compute_qa_mean, axis=1)

print("Data loaded:")
print(f"  Shape: {data.shape}")
print(f"  Sample:\n{data[['subject_id', 'split_pair', 'GA', 'abs_diff_sp', 'cnr_diff', 'QA_diff', 'QA_mean', 'stack_count']].head()}")

# ============================================================================
# REGRESSION FUNCTION
# ============================================================================

def fit_regression_model(data, outcome_col, predictor_cols, model_name=""):
    """Fit OLS regression and return results dict"""

    subset = data[predictor_cols + [outcome_col]].dropna()
    n = len(subset)

    if n < len(predictor_cols) + 3:
        print(f"  WARNING: n={n} is too small for {len(predictor_cols)} predictors")
        return None

    X = subset[predictor_cols]
    y = subset[outcome_col]

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - len(predictor_cols) - 1)

    mse = ss_res / (n - len(predictor_cols) - 1)
    ms_model = (ss_tot - ss_res) / len(predictor_cols)
    f_stat = ms_model / mse
    f_pvalue = 1 - stats.f.cdf(f_stat, len(predictor_cols), n - len(predictor_cols) - 1)

    residuals = y - y_pred
    residual_std_error = np.sqrt(mse)

    X_with_const = np.column_stack([np.ones(n), X])
    try:
        var_covar = mse * np.linalg.inv(X_with_const.T @ X_with_const)
        se_coef = np.sqrt(np.diag(var_covar))
    except:
        se_coef = np.ones(len(predictor_cols) + 1) * np.nan

    all_coef = np.concatenate([[model.intercept_], model.coef_])
    t_values = all_coef / se_coef
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), n - len(predictor_cols) - 1))

    coef_names = ['Intercept'] + list(predictor_cols)
    results = {
        'model_name': model_name,
        'n': n,
        'r2': r2,
        'adj_r2': adj_r2,
        'f_stat': f_stat,
        'f_pvalue': f_pvalue,
        'rmse': residual_std_error,
        'coefficients': {
            coef_names[i]: {
                'coef': all_coef[i],
                'se': se_coef[i],
                't': t_values[i],
                'p': p_values[i]
            }
            for i in range(len(coef_names))
        },
        'X': X,
        'y': y,
        'y_pred': y_pred,
        'residuals': residuals,
        'predictor_cols': predictor_cols
    }

    return results

# ============================================================================
# RESIDUALIZATION FUNCTION
# ============================================================================

def residualize(data, var_to_resid, var_reference, print_diagnostics=True):
    """Residualize var_to_resid against var_reference (remove reference effect)"""

    subset = data[[var_to_resid, var_reference]].dropna()
    X_ref = subset[[var_reference]]
    y_var = subset[var_to_resid]

    model = LinearRegression()
    model.fit(X_ref, y_var)
    y_pred = model.predict(X_ref)
    residuals = y_var - y_pred

    # Correlations
    r, p = stats.pearsonr(X_ref.values.flatten(), y_var)
    r2 = model.score(X_ref, y_var)

    if print_diagnostics:
        print(f"Residualizing: {var_to_resid} ~ {var_reference}")
        print(f"  Original correlation: r = {r:.4f}, p = {p:.4f}")
        print(f"  Regression R² = {r2:.4f}")
        print(f"  Slope = {model.coef_[0]:.4f} (p = {p:.4f})")
        print(f"  Residuals capture {(1-r2)*100:.1f}% unique variance")

    # Return residuals mapped back to original data index
    residuals_series = pd.Series(index=subset.index, data=residuals.values)
    full_residuals = pd.Series(index=data.index, dtype=float)
    full_residuals[residuals_series.index] = residuals_series

    return full_residuals

# ============================================================================
# MODEL 1: Vol_diff ~ GA + CNR_diff + QA_diff + stack_count
# ============================================================================

print("\n" + "="*80)
print("MODEL 1: Vol_diff ~ GA + CNR_diff + QA_diff + stack_count")
print("="*80)

results_m1 = {}
for split_pair in ['S1-S2', 'S3-S4']:
    subset = data[data['split_pair'] == split_pair]
    result = fit_regression_model(
        subset,
        outcome_col='abs_diff_sp',
        predictor_cols=['GA', 'cnr_diff', 'QA_diff', 'stack_count'],
        model_name=f"M1_{split_pair}"
    )
    if result:
        results_m1[split_pair] = result
        print(f"\n{split_pair}: N={result['n']}, R²={result['r2']:.3f}, Adj R²={result['adj_r2']:.3f}")
        print(f"  F={result['f_stat']:.2f}, p={result['f_pvalue']:.4f}")
        print("\n  Coefficients:")
        for var, coef_data in result['coefficients'].items():
            sig = "***" if coef_data['p'] < 0.001 else "**" if coef_data['p'] < 0.01 else "*" if coef_data['p'] < 0.05 else ""
            print(f"    {var:15s}: coef={coef_data['coef']:10.4f}, SE={coef_data['se']:10.4f}, t={coef_data['t']:7.3f}, p={coef_data['p']:.4f} {sig}")

# ============================================================================
# MODEL 2: Vol_diff ~ GA + CNR_diff + QA_mean + stack_count
# ============================================================================

print("\n" + "="*80)
print("MODEL 2: Vol_diff ~ GA + CNR_diff + QA_mean + stack_count")
print("="*80)

results_m2 = {}
for split_pair in ['S1-S2', 'S3-S4']:
    subset = data[data['split_pair'] == split_pair]
    result = fit_regression_model(
        subset,
        outcome_col='abs_diff_sp',
        predictor_cols=['GA', 'cnr_diff', 'QA_mean', 'stack_count'],
        model_name=f"M2_{split_pair}"
    )
    if result:
        results_m2[split_pair] = result
        print(f"\n{split_pair}: N={result['n']}, R²={result['r2']:.3f}, Adj R²={result['adj_r2']:.3f}")
        print(f"  F={result['f_stat']:.2f}, p={result['f_pvalue']:.4f}")
        print("\n  Coefficients:")
        for var, coef_data in result['coefficients'].items():
            sig = "***" if coef_data['p'] < 0.001 else "**" if coef_data['p'] < 0.01 else "*" if coef_data['p'] < 0.05 else ""
            print(f"    {var:15s}: coef={coef_data['coef']:10.4f}, SE={coef_data['se']:10.4f}, t={coef_data['t']:7.3f}, p={coef_data['p']:.4f} {sig}")

# ============================================================================
# MODEL 3: Residualized CNR_diff
# ============================================================================

print("\n" + "="*80)
print("MODEL 3: Vol_diff ~ GA + CNR_diff_resid + QA_diff + stack_count")
print("(CNR_diff_resid = CNR_diff with GA component removed)")
print("="*80)

# Residualize CNR_diff for S1-S2
print("\n--- Residualizing CNR_diff for S1-S2 ---")
data_resid_s12 = data[data['split_pair'] == 'S1-S2'].copy()
data_resid_s12['cnr_diff_resid'] = residualize(
    data_resid_s12, 'cnr_diff', 'GA', print_diagnostics=True
)

# Residualize CNR_diff for S3-S4
print("\n--- Residualizing CNR_diff for S3-S4 ---")
data_resid_s34 = data[data['split_pair'] == 'S3-S4'].copy()
data_resid_s34['cnr_diff_resid'] = residualize(
    data_resid_s34, 'cnr_diff', 'GA', print_diagnostics=True
)

results_m3 = {}
for split_pair, data_resid in [('S1-S2', data_resid_s12), ('S3-S4', data_resid_s34)]:
    result = fit_regression_model(
        data_resid,
        outcome_col='abs_diff_sp',
        predictor_cols=['GA', 'cnr_diff_resid', 'QA_diff', 'stack_count'],
        model_name=f"M3_{split_pair}"
    )
    if result:
        results_m3[split_pair] = result
        print(f"\n{split_pair}: N={result['n']}, R²={result['r2']:.3f}, Adj R²={result['adj_r2']:.3f}")
        print(f"  F={result['f_stat']:.2f}, p={result['f_pvalue']:.4f}")
        print("\n  Coefficients:")
        for var, coef_data in result['coefficients'].items():
            sig = "***" if coef_data['p'] < 0.001 else "**" if coef_data['p'] < 0.01 else "*" if coef_data['p'] < 0.05 else ""
            print(f"    {var:15s}: coef={coef_data['coef']:10.4f}, SE={coef_data['se']:10.4f}, t={coef_data['t']:7.3f}, p={coef_data['p']:.4f} {sig}")

# ============================================================================
# SAVE TABLES
# ============================================================================

os.makedirs('../../assets', exist_ok=True)

# Model summary table
summary_data = []
for model_num, results_dict in [(1, results_m1), (2, results_m2), (3, results_m3)]:
    for split_pair, result in results_dict.items():
        summary_data.append({
            'Model': f'M{model_num}',
            'Split': split_pair,
            'N': result['n'],
            'R_squared': round(result['r2'], 3),
            'Adj_R_squared': round(result['adj_r2'], 3),
            'F_statistic': round(result['f_stat'], 2),
            'F_p_value': round(result['f_pvalue'], 4),
            'RMSE': round(result['rmse'], 2)
        })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('../../assets/cnr_multivariate_summary.csv', index=False)
print("\n✓ Summary saved to cnr_multivariate_summary.csv")
print(summary_df.to_string(index=False))

# Coefficient table
coef_table = []
for model_num, model_name, results_dict in [
    (1, "M1_base", results_m1),
    (2, "M2_qa_mean", results_m2),
    (3, "M3_resid", results_m3)
]:
    for split_pair, result in results_dict.items():
        row = {
            'Model': model_name,
            'Split': split_pair,
            'N': result['n'],
            'R²': f"{result['r2']:.3f}",
            'Adj_R²': f"{result['adj_r2']:.3f}",
            'F': f"{result['f_stat']:.2f}",
            'F_p': f"{result['f_pvalue']:.4f}",
        }
        for var, coef_data in result['coefficients'].items():
            sig = "***" if coef_data['p'] < 0.001 else "**" if coef_data['p'] < 0.01 else "*" if coef_data['p'] < 0.05 else ""
            row[f"{var}_coef"] = f"{coef_data['coef']:.4f}"
            row[f"{var}_SE"] = f"{coef_data['se']:.4f}"
            row[f"{var}_p"] = f"{coef_data['p']:.4f}{sig}"
        coef_table.append(row)

coef_df = pd.DataFrame(coef_table)
coef_df.to_csv('../../assets/cnr_multivariate_coefficients.csv', index=False)
print("\n✓ Coefficients saved to cnr_multivariate_coefficients.csv")

# Save for plotting
import pickle
all_results = {
    'm1': results_m1,
    'm2': results_m2,
    'm3': results_m3,
    'data': data,
    'data_resid_s12': data_resid_s12,
    'data_resid_s34': data_resid_s34
}

with open('../../assets/cnr_multivariate_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)

print("\n✓ Full results saved to cnr_multivariate_results.pkl for plotting")
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
