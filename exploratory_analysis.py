# Imports
import os, sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from warnings import filterwarnings
filterwarnings('ignore')

# Ensure project root is on sys.path so we can import scripts.utils
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.utils import (
    load_csv,
    save_table,
    correlation_heatmap,
    labeled_boxplot,
    labeled_hist,
)

DATA_DIR = os.path.join('.', 'outputs')
FIG_DIR = os.path.join('.', 'outputs', 'figures')
TABLE_DIR = os.path.join('.', 'outputs', 'tables')

clean1_path = os.path.join(DATA_DIR, 'cleaned_dataset1.csv')
clean2_path = os.path.join(DATA_DIR, 'cleaned_dataset2.csv')

df1 = pd.read_csv(clean1_path)
df2 = pd.read_csv(clean2_path)

# Merge if same keys exist; else keep separate
# Try a safe concat for EDA scope
eda_df = pd.concat([df1.assign(source='dataset1'), df2.assign(source='dataset2')], ignore_index=True, sort=False)
print('Combined EDA shape:', eda_df.shape)


# Summary statistics for numeric columns
numeric_desc = eda_df.select_dtypes(include='number').describe().T
save_table(numeric_desc, os.path.join(TABLE_DIR, 'eda_numeric_summary.csv'))

# Frequency distributions
freq_risk = eda_df['risk'].value_counts(dropna=False).rename_axis('risk').to_frame('count') if 'risk' in eda_df.columns else pd.DataFrame()
freq_reward = eda_df['reward'].value_counts(dropna=False).rename_axis('reward').to_frame('count') if 'reward' in eda_df.columns else pd.DataFrame()
freq_season = eda_df['season'].value_counts(dropna=False).rename_axis('season').to_frame('count') if 'season' in eda_df.columns else pd.DataFrame()

if not freq_risk.empty:
    save_table(freq_risk, os.path.join(TABLE_DIR, 'freq_risk.csv'))
if not freq_reward.empty:
    save_table(freq_reward, os.path.join(TABLE_DIR, 'freq_reward.csv'))
if not freq_season.empty:
    save_table(freq_season, os.path.join(TABLE_DIR, 'freq_season.csv'))

numeric_desc.head()



fig = correlation_heatmap(eda_df, title='Figure 1 – Correlation Heatmap')
fig_path = os.path.join(FIG_DIR, 'figure_01_correlation_heatmap.png')
from scripts.utils import save_figure
save_figure(fig, fig_path)
fig_path


# Boxplots for numeric columns by source
if set(['risk','seconds_after_rat_arrival']).issubset(eda_df.columns):
    # Bin seconds_after_rat_arrival into categories to use as x-axis
    bins = [-np.inf, 0, 30, 60, 120, np.inf]
    labels = ['<=0s','0-30s','30-60s','60-120s','>120s']
    import numpy as np
    eda_df['rat_arrival_bin'] = pd.cut(eda_df['seconds_after_rat_arrival'], bins=bins, labels=labels)
    fig = labeled_boxplot(eda_df.dropna(subset=['rat_arrival_bin','risk']), x='rat_arrival_bin', y='risk', title='Figure 2 – Risk vs. Time After Rat Arrival', xlabel='Seconds after rat arrival (binned)', ylabel='Risk')
    save_figure(fig, os.path.join(FIG_DIR, 'figure_02_risk_vs_seconds_after_rat_arrival.png'))
else:
    print('Required columns for boxplot not found.')


# Histograms for numeric columns
if 'bat_landing_to_food' in eda_df.columns:
    fig = labeled_hist(eda_df['bat_landing_to_food'], bins=30, title='Figure 3 – Distribution of Bat Landing to Food', xlabel='Seconds')
    save_figure(fig, os.path.join(FIG_DIR, 'figure_03_hist_bat_landing_to_food.png'))
else:
    print('bat_landing_to_food column not found.')


from scripts.utils import labeled_barplot

if 'season' in eda_df.columns:
    for var in ['risk','reward','bat_landing_to_food','rat_arrival_number']:
        if var in eda_df.columns:
            fig = labeled_barplot(eda_df.dropna(subset=['season', var]), x='season', y=var, title=f'Figure 4 – Average {var} by Season')
            save_figure(fig, os.path.join(FIG_DIR, f'figure_04_avg_{var}_by_season.png'))
else:
    print('season column not found.')
