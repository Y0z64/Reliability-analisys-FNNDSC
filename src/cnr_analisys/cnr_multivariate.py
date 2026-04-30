"""
CNR Multivariate Analysis: Volume Instability Prediction
Models volume differences using CNR and related predictors
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
import seaborn as sns
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

qa_cols = ['subject_id', 'session_id', 'QA_S1', 'QA_S2', 'QA_S3', 'QA_S4', 'QA12_mean', 'QA34_mean']
data = split_pair_diffs.merge(
    infodump_df[qa_cols],
    on=['subject_id', 'session_id'],
    how='left'
)
data['QA_diff'] = data.apply(compute_qa_diff, axis=1)

# ============================================================================
# REGRESSION FUNCTION
# ============================================================================

def fit_regression_model(data, outcome_col, predictor_cols, model_name=""):
    """Fit OLS regression and return results dict"""

    # Select complete cases
    subset = data[predictor_cols + [outcome_col]].dropna()
    n = len(subset)

    if n < len(predictor_cols) + 3:
        print(f"  WARNING: n={n} is too small for {len(predictor_cols)} predictors")
        return None

    X = subset[predictor_cols]
    y = subset[outcome_col]

    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Calculate statistics
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - len(predictor_cols) - 1)

    # F-statistic
    mse = ss_res / (n - len(predictor_cols) - 1)
    ms_model = (ss_tot - ss_res) / len(predictor_cols)
    f_stat = ms_model / mse
    f_pvalue = 1 - stats.f.cdf(f_stat, len(predictor_cols), n - len(predictor_cols) - 1)

    # Standard errors and t-values
    residuals = y - y_pred
    residual_std_error = np.sqrt(mse)

    # Design matrix for SE calculation
    X_with_const = np.column_stack([np.ones(n), X])
    try:
        var_covar = mse * np.linalg.inv(X_with_const.T @ X_with_const)
        se_coef = np.sqrt(np.diag(var_covar))
    except:
        se_coef = np.ones(len(predictor_cols) + 1) * np.nan

    # T-values and p-values
    all_coef = np.concatenate([[model.intercept_], model.coef_])
    t_values = all_coef / se_coef
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), n - len(predictor_cols) - 1))

    # Build results dict
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
# MODEL 1: CNR_DIFF + GA
# ============================================================================

print("\n" + "="*80)
print("MODEL 1: Volume ~ GA + CNR_diff")
print("="*80)

results_cnr_diff = {}
for split_pair in ['S1-S2', 'S3-S4']:
    subset = data[data['split_pair'] == split_pair]
    result = fit_regression_model(
        subset,
        outcome_col='abs_diff_sp',
        predictor_cols=['GA', 'cnr_diff'],
        model_name=f"CNR_diff_{split_pair}"
    )
    if result:
        results_cnr_diff[split_pair] = result
        print(f"\n{split_pair}: N={result['n']}, R²={result['r2']:.3f}, F={result['f_stat']:.2f}, p={result['f_pvalue']:.4f}")
        for var, coef_data in result['coefficients'].items():
            sig = "***" if coef_data['p'] < 0.001 else "**" if coef_data['p'] < 0.01 else "*" if coef_data['p'] < 0.05 else ""
            print(f"  {var:12s}: coef={coef_data['coef']:7.3f}, SE={coef_data['se']:7.3f}, p={coef_data['p']:.4f} {sig}")

# ============================================================================
# MODEL 2: CNR_DIFF + GA + QA_diff
# ============================================================================

print("\n" + "="*80)
print("MODEL 2: Volume ~ GA + CNR_diff + QA_diff")
print("="*80)

results_cnr_qa = {}
for split_pair in ['S1-S2', 'S3-S4']:
    subset = data[data['split_pair'] == split_pair]
    result = fit_regression_model(
        subset,
        outcome_col='abs_diff_sp',
        predictor_cols=['GA', 'cnr_diff', 'QA_diff'],
        model_name=f"CNR_QA_{split_pair}"
    )
    if result:
        results_cnr_qa[split_pair] = result
        print(f"\n{split_pair}: N={result['n']}, R²={result['r2']:.3f}, F={result['f_stat']:.2f}, p={result['f_pvalue']:.4f}")
        for var, coef_data in result['coefficients'].items():
            sig = "***" if coef_data['p'] < 0.001 else "**" if coef_data['p'] < 0.01 else "*" if coef_data['p'] < 0.05 else ""
            print(f"  {var:12s}: coef={coef_data['coef']:7.3f}, SE={coef_data['se']:7.3f}, p={coef_data['p']:.4f} {sig}")

# ============================================================================
# MODEL 3: CNR_MEAN + GA
# ============================================================================

print("\n" + "="*80)
print("MODEL 3: Volume ~ GA + CNR_mean")
print("="*80)

results_cnr_mean = {}
for split_pair in ['S1-S2', 'S3-S4']:
    subset = data[data['split_pair'] == split_pair]
    result = fit_regression_model(
        subset,
        outcome_col='abs_diff_sp',
        predictor_cols=['GA', 'cnr_mean'],
        model_name=f"CNR_mean_{split_pair}"
    )
    if result:
        results_cnr_mean[split_pair] = result
        print(f"\n{split_pair}: N={result['n']}, R²={result['r2']:.3f}, F={result['f_stat']:.2f}, p={result['f_pvalue']:.4f}")
        for var, coef_data in result['coefficients'].items():
            sig = "***" if coef_data['p'] < 0.001 else "**" if coef_data['p'] < 0.01 else "*" if coef_data['p'] < 0.05 else ""
            print(f"  {var:12s}: coef={coef_data['coef']:7.3f}, SE={coef_data['se']:7.3f}, p={coef_data['p']:.4f} {sig}")

# ============================================================================
# SAVE TABLES FOR OBSIDIAN
# ============================================================================

os.makedirs('../../assets', exist_ok=True)

# Compile coefficient table
coef_table = []
for model_name, results_dict in [
    ("CNR_diff", results_cnr_diff),
    ("CNR_QA", results_cnr_qa),
    ("CNR_mean", results_cnr_mean)
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
        # Add coefficients
        for var, coef_data in result['coefficients'].items():
            sig = "***" if coef_data['p'] < 0.001 else "**" if coef_data['p'] < 0.01 else "*" if coef_data['p'] < 0.05 else ""
            row[f"{var}_coef"] = f"{coef_data['coef']:.4f}"
            row[f"{var}_p"] = f"{coef_data['p']:.4f}{sig}"
        coef_table.append(row)

coef_df = pd.DataFrame(coef_table)
coef_df.to_csv('../../assets/cnr_regression_coefficients.csv', index=False)
print("\n✓ Coefficients saved to cnr_regression_coefficients.csv")

# Model summary table
summary_data = []
for model_name, results_dict in [
    ("CNR_diff", results_cnr_diff),
    ("CNR_QA", results_cnr_qa),
    ("CNR_mean", results_cnr_mean)
]:
    for split_pair, result in results_dict.items():
        summary_data.append({
            'Model': model_name,
            'Split': split_pair,
            'N': result['n'],
            'R_squared': round(result['r2'], 3),
            'Adj_R_squared': round(result['adj_r2'], 3),
            'F_statistic': round(result['f_stat'], 2),
            'F_p_value': round(result['f_pvalue'], 4),
            'RMSE': round(result['rmse'], 2)
        })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('../../assets/cnr_model_summary.csv', index=False)
print("✓ Summary saved to cnr_model_summary.csv")

print("\n" + summary_df.to_string(index=False))

# ============================================================================
# SAVE FOR LATER PLOTTING
# ============================================================================

import pickle

all_results = {
    'cnr_diff': results_cnr_diff,
    'cnr_qa': results_cnr_qa,
    'cnr_mean': results_cnr_mean,
    'data': data
}

with open('../../assets/cnr_regression_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)

print("\n✓ Full results saved to cnr_regression_results.pkl for plotting")

