"""
CNR Multivariate Analysis: Publication-quality plots
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pickle
from sklearn.linear_model import LinearRegression

# Load results
with open('../../assets/cnr_regression_results.pkl', 'rb') as f:
    all_results = pickle.load(f)

results_cnr_diff = all_results['cnr_diff']
results_cnr_qa = all_results['cnr_qa']
results_cnr_mean = all_results['cnr_mean']
data = all_results['data']

# ============================================================================
# PLOT 1: Predicted vs Actual - CNR_diff Model
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (split_pair, ax) in enumerate(zip(['S1-S2', 'S3-S4'], axes)):
    result = results_cnr_diff[split_pair]

    # Plot actual vs predicted
    ax.scatter(result['y'], result['y_pred'], s=100, alpha=0.6, color='steelblue', edgecolors='black', linewidth=0.5)

    # Perfect prediction line
    min_val = min(result['y'].min(), result['y_pred'].min())
    max_val = max(result['y'].max(), result['y_pred'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.7, label='Perfect fit')

    # Regression line of actual vs predicted
    z = np.polyfit(result['y'], result['y_pred'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(min_val, max_val, 100)
    ax.plot(x_line, p(x_line), 'g-', linewidth=2, alpha=0.7, label='Fitted line')

    ax.set_xlabel('Actual Volume Difference (mm³)', fontsize=11)
    ax.set_ylabel('Predicted Volume Difference (mm³)', fontsize=11)
    ax.set_title(f'{split_pair}: Model Predictions (R²={result["r2"]:.3f})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # Add stats text
    rmse = np.sqrt(np.mean(result['residuals']**2))
    ax.text(0.05, 0.95, f"N={result['n']}\nRMSE={rmse:.1f}\nR²={result['r2']:.3f}",
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('../../assets/cnr_pred_actual_01.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved cnr_pred_actual_01.png")

# ============================================================================
# PLOT 2: Residuals - CNR_diff Model
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (split_pair, result) in enumerate(results_cnr_diff.items()):
    # Residuals vs Fitted
    ax = axes[0, idx]
    ax.scatter(result['y_pred'], result['residuals'], s=100, alpha=0.6, color='steelblue', edgecolors='black', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Fitted Values (mm³)', fontsize=10)
    ax.set_ylabel('Residuals (mm³)', fontsize=10)
    ax.set_title(f'{split_pair}: Residuals vs Fitted', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Q-Q plot
    ax = axes[1, idx]
    stats.probplot(result['residuals'], dist="norm", plot=ax)
    ax.set_title(f'{split_pair}: Q-Q Plot', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../../assets/cnr_residuals_02.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved cnr_residuals_02.png")

# ============================================================================
# PLOT 3: Coefficient Comparison Across Models
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, split_pair in enumerate(['S1-S2', 'S3-S4']):
    ax = axes[idx]

    # Extract GA coefficient and 95% CI for each model
    models = []
    ga_coefs = []
    ga_errors = []

    for model_name, results_dict in [('CNR_diff', results_cnr_diff), ('CNR_QA', results_cnr_qa), ('CNR_mean', results_cnr_mean)]:
        if split_pair in results_dict:
            result = results_dict[split_pair]
            ga_coef = result['coefficients']['GA']['coef']
            ga_se = result['coefficients']['GA']['se']
            ga_ci = 1.96 * ga_se  # 95% CI

            models.append(model_name)
            ga_coefs.append(ga_coef)
            ga_errors.append(ga_ci)

    # Plot
    x_pos = np.arange(len(models))
    ax.bar(x_pos, ga_coefs, yerr=ga_errors, capsize=5, alpha=0.7, color=['#e74c3c', '#3498db', '#2ecc71'],
           edgecolor='black', linewidth=1)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel('GA Coefficient ± 95% CI', fontsize=11)
    ax.set_title(f'{split_pair}: GA Effect Across Models', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../../assets/cnr_coef_comparison_03.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved cnr_coef_comparison_03.png")

# ============================================================================
# PLOT 4: Model Fit Comparison (R² and AIC)
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

summary_data = []
for model_name, results_dict in [('CNR_diff', results_cnr_diff), ('CNR_QA', results_cnr_qa), ('CNR_mean', results_cnr_mean)]:
    for split_pair, result in results_dict.items():
        # Calculate AIC
        n = result['n']
        k = len(result['predictor_cols']) + 1  # +1 for intercept
        ss_res = np.sum(result['residuals']**2)
        aic = n * np.log(ss_res/n) + 2*k

        summary_data.append({
            'Model': model_name,
            'Split': split_pair,
            'R²': result['r2'],
            'AIC': aic,
            'F_p': result['f_pvalue']
        })

summary_df = pd.DataFrame(summary_data)

# R² plot
ax = axes[0]
for split_pair in ['S1-S2', 'S3-S4']:
    subset = summary_df[summary_df['Split'] == split_pair]
    color = '#e74c3c' if split_pair == 'S1-S2' else '#3498db'
    ax.plot(subset.index % 3, subset['R²'].values, 'o-', color=color, linewidth=2, markersize=10, label=split_pair)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['CNR_diff', 'CNR_QA', 'CNR_mean'])
ax.set_ylabel('R² (Variance Explained)', fontsize=11)
ax.set_title('Model Fit: R² Comparison', fontsize=12, fontweight='bold')
ax.set_ylim([0, 0.6])
ax.grid(True, alpha=0.3, axis='y')
ax.legend(loc='best')

# AIC plot (lower is better)
ax = axes[1]
for split_pair in ['S1-S2', 'S3-S4']:
    subset = summary_df[summary_df['Split'] == split_pair]
    color = '#e74c3c' if split_pair == 'S1-S2' else '#3498db'
    ax.plot(subset.index % 3, subset['AIC'].values, 's-', color=color, linewidth=2, markersize=10, label=split_pair)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['CNR_diff', 'CNR_QA', 'CNR_mean'])
ax.set_ylabel('AIC (lower is better)', fontsize=11)
ax.set_title('Model Fit: AIC Comparison', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.legend(loc='best')

plt.tight_layout()
plt.savefig('../../assets/cnr_model_fit_04.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved cnr_model_fit_04.png")

# ============================================================================
# PLOT 5: Partial Regression Plot - GA effect (CNR_QA model, best fit)
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (split_pair, ax) in enumerate(zip(['S1-S2', 'S3-S4'], axes)):
    result = results_cnr_qa[split_pair]

    # Get data
    X = result['X']
    y = result['y']

    # Residuals of y after removing other predictors
    X_others = X[['cnr_diff', 'QA_diff']]
    y_resid = y - LinearRegression().fit(X_others, y).predict(X_others)
    X_ga_resid = X['GA'] - LinearRegression().fit(X_others, X['GA']).predict(X_others)

    # Scatter
    ax.scatter(X['GA'], y, s=100, alpha=0.5, color='lightgray', label='Raw values')

    # Fit line through GA
    z = np.polyfit(X['GA'], y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(X['GA'].min(), X['GA'].max(), 100)
    ax.plot(x_line, p(x_line), 'b-', linewidth=2, label='Marginal relationship')

    # Get partial regression info
    ga_coef = result['coefficients']['GA']['coef']
    ga_p = result['coefficients']['GA']['p']

    ax.set_xlabel('Gestational Age (weeks)', fontsize=11)
    ax.set_ylabel('Volume Difference (mm³)', fontsize=11)
    ax.set_title(f'{split_pair}: GA Effect on Volume\n(from CNR_QA model)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # Stats
    sig = "***" if ga_p < 0.001 else "**" if ga_p < 0.01 else "*" if ga_p < 0.05 else "ns"
    ax.text(0.05, 0.95, f"β={ga_coef:.2f}\np={ga_p:.4f} {sig}\nR²={result['r2']:.3f}",
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('../../assets/cnr_ga_effect_05.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved cnr_ga_effect_05.png")

print("\n✓ All plots saved to assets folder")

