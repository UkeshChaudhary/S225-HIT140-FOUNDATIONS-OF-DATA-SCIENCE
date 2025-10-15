# Imports and configuration
import os
import pandas as pd
from scripts.utils import (
    load_csv,
    save_csv,
    coerce_datetime,
    coerce_numeric,
    add_engineered_columns,
    basic_info,
    handle_missing_values,
)


# load Datasets
DATA_DIR = os.path.join('.', 'data')
OUTPUTS_DIR = os.path.join('.', 'outputs')
FIG_DIR = os.path.join(OUTPUTS_DIR, 'figures')
TABLE_DIR = os.path.join(OUTPUTS_DIR, 'tables')

# Load raw datasets
path1 = os.path.join(DATA_DIR, 'dataset1.csv')
path2 = os.path.join(DATA_DIR, 'dataset2.csv')

df1_raw = load_csv(path1)
df2_raw = load_csv(path2)

print('Dataset1 shape:', df1_raw.shape)
print('Dataset2 shape:', df2_raw.shape)

df1_raw.head()

# Display info, missing values, and data types
print('\nDataset1 info:')
df1_dtypes, df1_missing, df1_desc = basic_info(df1_raw)
print(df1_dtypes.head())
print('\nDataset1 missing counts:\n', df1_missing[df1_missing > 0].sort_values(ascending=False))

print('\nDataset2 info:')
df2_dtypes, df2_missing, df2_desc = basic_info(df2_raw)
print(df2_dtypes.head())
print('\nDataset2 missing counts:\n', df2_missing[df2_missing > 0].sort_values(ascending=False))

df1_desc.head()

# Candidate datetime columns
candidate_time_cols = [
    'rat_period_start', 'rat_period_end',
    'observation_start', 'observation_end',
]

# Coerce if present
df1 = coerce_datetime(df1_raw.copy(), candidate_time_cols)
df2 = coerce_datetime(df2_raw.copy(), candidate_time_cols)

# Also coerce numerics commonly used later
numeric_candidates = [
    'bat_landing_to_food', 'hours_after_sunset', 'seconds_after_rat_arrival',
    'risk', 'reward', 'rat_arrival_number'
]

# Coerce if present
df1 = coerce_numeric(df1, numeric_candidates)
df2 = coerce_numeric(df2, numeric_candidates)

print('Datetime coercion done.')

# Handle missing values - simple strategies for now
df1_clean = handle_missing_values(df1, strategy='median')
df2_clean = handle_missing_values(df2, strategy='median')

print('Missing-value handling complete.')

# Add engineered columns
df1_engineered = add_engineered_columns(df1_clean)
df2_engineered = add_engineered_columns(df2_clean)

# Quick spot-check of new fields
df1_engineered[[
    c for c in ['rat_presence_duration', 'response_delay', 'is_night'] if c in df1_engineered.columns
]].head()

# Save cleaned datasets

clean1_path = os.path.join('.', 'outputs', 'cleaned_dataset1.csv')
clean2_path = os.path.join('.', 'outputs', 'cleaned_dataset2.csv')

save_csv(df1_engineered, clean1_path)
save_csv(df2_engineered, clean2_path)

print('Saved cleaned datasets to:', clean1_path, 'and', clean2_path)
