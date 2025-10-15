# Imports
import os, sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure project root is on sys.path so we can import scripts.utils
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.utils import (
    load_csv,
    save_table,
    save_figure,
    pearson_correlation,
    logistic_regression,
    correlation_heatmap,
    labeled_boxplot,
)

OUTPUTS_DIR = os.path.join('.', 'outputs')
FIG_DIR = os.path.join(OUTPUTS_DIR, 'figures')
TABLE_DIR = os.path.join(OUTPUTS_DIR, 'tables')

clean1_path = os.path.join('.', 'outputs', 'cleaned_dataset1.csv')
clean2_path = os.path.join('.', 'outputs', 'cleaned_dataset2.csv')

df1 = pd.read_csv(clean1_path)
df2 = pd.read_csv(clean2_path)

# Combine for broader analysis
A_df = pd.concat([df1.assign(source='dataset1'), df2.assign(source='dataset2')], ignore_index=True, sort=False)

# Ensure key columns exist
required_cols = ['risk','reward','seconds_after_rat_arrival','rat_presence_duration']
for c in required_cols:
    if c not in A_df.columns:
        A_df[c] = pd.NA

print('Investigation A dataset shape:', A_df.shape)


# Correlations
pairs = [('risk','seconds_after_rat_arrival'), ('risk','rat_presence_duration'), ('reward','seconds_after_rat_arrival')]
rows = []
for x, y in pairs:
    if x in A_df.columns and y in A_df.columns:
        r, p = pearson_correlation(A_df[x], A_df[y])
        rows.append({'x': x, 'y': y, 'r': r, 'p_value': p})

corr_table = pd.DataFrame(rows)
save_table(corr_table.set_index(['x','y']), os.path.join(TABLE_DIR, 'A_correlations.csv'))
corr_table


# Logistic regression: risk (binary) vs predictors
A_model_df = A_df.copy()
if 'risk' in A_model_df.columns:
    # Coerce risk to binary for logistic model
    A_model_df['risk_bin'] = (pd.to_numeric(A_model_df['risk'], errors='coerce') > 0).astype('Int64')

# Drop rows missing predictors
predictors = [c for c in ['seconds_after_rat_arrival','rat_presence_duration'] if c in A_model_df.columns]
model_formula = 'risk_bin ~ ' + ' + '.join(predictors)
A_model_df = A_model_df.dropna(subset=['risk_bin'] + predictors)

if A_model_df.shape[0] > 0 and len(predictors) > 0:
    model = logistic_regression(A_model_df, model_formula)
    summ = model.summary2().tables[1]
    save_table(summ, os.path.join(TABLE_DIR, 'A_logistic_regression_coeffs.csv'))
    print(model.summary())
else:
    print('Not enough data/columns to fit logistic regression.')



# Scatter + regression line: risk vs reward
if set(['risk','reward']).issubset(A_df.columns):
    fig, ax = plt.subplots(figsize=(7,5))
    sns.regplot(data=A_df, x='reward', y='risk', scatter_kws={'alpha':0.4}, ax=ax)
    ax.set_title('Figure A1 – Risk vs Reward with Regression Line')
    save_figure(fig, os.path.join(FIG_DIR, 'figure_A1_risk_vs_reward_reg.png'))
else:
    print('risk or reward not available for scatter/regression.')

# Heatmap of behavioural correlations
fig = correlation_heatmap(A_df, title='Figure A2 – Behavioural Correlations')
save_figure(fig, os.path.join(FIG_DIR, 'figure_A2_behavioural_correlations.png'))

# Boxplots: high/low rat presence vs risk
if 'rat_presence_duration' in A_df.columns and 'risk' in A_df.columns:
    cut = pd.qcut(pd.to_numeric(A_df['rat_presence_duration'], errors='coerce'), q=2, labels=['Low','High'])
    A_df_plot = A_df.copy()
    A_df_plot['rat_presence_level'] = cut
    fig = labeled_boxplot(A_df_plot.dropna(subset=['rat_presence_level','risk']), x='rat_presence_level', y='risk', title='Figure A3 – Risk by Rat Presence Level', xlabel='Rat Presence Level', ylabel='Risk')
    save_figure(fig, os.path.join(FIG_DIR, 'figure_A3_risk_by_rat_presence_level.png'))
else:
    print('Missing columns for boxplot of rat presence vs risk.')
