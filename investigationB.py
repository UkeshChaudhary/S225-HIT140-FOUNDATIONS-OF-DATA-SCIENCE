# Imports
import os, sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from warnings import filterwarnings
filterwarnings('ignore')

# Ensure project root is on sys.path so we can import scripts.utils
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.utils import (
    save_table,
    save_figure,
    labeled_barplot,
    labeled_violinplot,
    labeled_lineplot,
    t_test_independent,
)

OUTPUTS_DIR = os.path.join('.', 'outputs')
FIG_DIR = os.path.join(OUTPUTS_DIR, 'figures')
TABLE_DIR = os.path.join(OUTPUTS_DIR, 'tables')

clean1_path = os.path.join('.', 'outputs', 'cleaned_dataset1.csv')
clean2_path = os.path.join('.', 'outputs', 'cleaned_dataset2.csv')

df1 = pd.read_csv(clean1_path)
df2 = pd.read_csv(clean2_path)

B_df = pd.concat([df1.assign(source='dataset1'), df2.assign(source='dataset2')], ignore_index=True, sort=False)
print('Investigation B dataset shape:', B_df.shape)


group_vars = [v for v in ['risk','reward','bat_landing_to_food','rat_arrival_number'] if v in B_df.columns]

if 'season' in B_df.columns and group_vars:
    seasonal_means = B_df.groupby('season')[group_vars].mean(numeric_only=True)
    save_table(seasonal_means, os.path.join(TABLE_DIR, 'B_seasonal_means.csv'))
    seasonal_means
else:
    print('Missing season column or variables to group.')


if 'season' in B_df.columns and 'risk' in B_df.columns:
    winter = B_df[B_df['season'].str.lower()== 'winter'] if B_df['season'].dtype==object else B_df[B_df['season']== 'winter']
    spring = B_df[B_df['season'].str.lower()== 'spring'] if B_df['season'].dtype==object else B_df[B_df['season']== 'spring']
    if not winter.empty and not spring.empty:
        from scripts.utils import t_test_independent
        t_stat, p_val = t_test_independent(winter['risk'], spring['risk'])
        test_res = pd.DataFrame({'t_statistic':[t_stat], 'p_value':[p_val]}, index=['winter_vs_spring_risk'])
        save_table(test_res, os.path.join(TABLE_DIR, 'B_ttest_risk_winter_vs_spring.csv'))
        test_res
    else:
        print('Winter or Spring subset empty for risk test.')
else:
    print('Missing season or risk column for t-test.')



# Barplot – average risk by season
if set(['season','risk']).issubset(B_df.columns):
    fig = labeled_barplot(B_df.dropna(subset=['season','risk']), x='season', y='risk', title='Figure B1 – Average Risk by Season')
    save_figure(fig, os.path.join(FIG_DIR, 'figure_B1_avg_risk_by_season.png'))
else:
    print('Missing columns for risk barplot.')

# Violinplot – bat_landing_to_food by season
if set(['season','bat_landing_to_food']).issubset(B_df.columns):
    fig = labeled_violinplot(B_df.dropna(subset=['season','bat_landing_to_food']), x='season', y='bat_landing_to_food', title='Figure B2 – Response Delay by Season')
    save_figure(fig, os.path.join(FIG_DIR, 'figure_B2_violin_bat_landing_to_food_by_season.png'))
else:
    print('Missing columns for violin plot.')

# Lineplot – month vs rat_arrival_number
# Attempt to parse month if available
def infer_month(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ['observation_start','rat_period_start']:
        if col in out.columns:
            ts = pd.to_datetime(out[col], errors='coerce')
            if ts.notna().any():
                out['month'] = ts.dt.month
                return out
    return out

B_df_m = infer_month(B_df)
if set(['month','rat_arrival_number']).issubset(B_df_m.columns):
    fig = labeled_lineplot(B_df_m.dropna(subset=['month','rat_arrival_number']).sort_values('month'), x='month', y='rat_arrival_number', title='Figure B3 – Rat Arrivals by Month')
    save_figure(fig, os.path.join(FIG_DIR, 'figure_B3_line_rat_arrival_number_by_month.png'))
else:
    print('Missing month or rat_arrival_number for line plot.')



