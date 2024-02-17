# %%
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import trim_mean

import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.io as pio
from plotly.subplots import make_subplots


import sys
sys.path.insert(0, '../')


import RepeatedMeasuresModel

# %%
# plotting parameters
grey = "#21201F"
green = "#9AC529"
lblue = "#42B9B2"
pink = "#DE237B"
orange = "#F38A31"


# %% [markdown]
# # Load Data
# Loads data from the computed markers. From `Data` directory
data_path = "Data/"
results_path = "Results/Multivariate_all_against_all/"
fig_path = results_path + "Figs/"

df = pd.read_csv(os.path.join(data_path, 'all_markers.csv'), index_col = 0)

# %%
markers = ['wSMI_1', 'wSMI_2', 'wSMI_4', 'wSMI_8', 'p_e_1', 'p_e_2',
       'p_e_4', 'p_e_8', 'k', 'se','msf', 'sef90', 'sef95', 'b', 'b_n', 'g',
       'g_n', 't', 't_n', 'd', 'd_n', 'a_n', 'a', 'CNV', 'P1', 'P3a', 'P3b',]

erps =['CNV', 'P1', 'P3a', 'P3b']

df_markers = (df
              .dropna()
              .query("stimuli == 'go'") # only go trials
              .query("correct == 'correct'") #only correct trials
              .query('prev_trial < 6') # only last 5 trials before each probe. 
              .drop(['stimuli', 'correct', 'prev_trial', 'label', 'events',  'epoch_type', 'preproc', 'ft', 'ft_n'], axis = 1) # drop unnecessary columns
              .query("mind in ['on-task','dMW', 'sMW']") # only mind wandering and on-task trials
            #   .groupby(['segment', 'participant']).filter(lambda x: len(x) > 1) # drop participants with less than 2 trials per segment
             )


comparisons = ['on-task_vs_mw','on-task_vs_dMW', 'on-task_vs_sMW', 'dMW_vs_sMW']

def preprocess_data(df_markers, markers, probe_type, comparison=None, only_full_participants=False, latex_names=False, results_path=None):
    # Filtering and grouping
    df = df_markers.query(f"probe == '{probe_type}'")

    # Adjust mind categories based on comparison type
    if comparison:
        if comparison == 'on-task_vs_mw':
            df['mind_category'] = df['mind'].replace({'dMW': 'mw', 'sMW': 'mw'})
        elif comparison in ['on-task_vs_dMW', 'on-task_vs_sMW', 'dMW_vs_sMW']:
            mind_types = comparison.split('_vs_')
            df = df[df['mind'].isin(mind_types)]
            df['mind_category'] = df['mind']
    else:
        df['mind_category'] = df['mind']

    # Aggregation dictionary
    agg_dict = {k: [apply_trim_mean,'std'] for k in markers}
    agg_dict.update({k: 'first' for k in df.drop(markers, axis=1).columns})
    df = df.groupby(['segment', 'participant'], as_index=False).agg(agg_dict)
    # df = df.groupby(['mind_category', 'participant'], as_index=False).agg(agg_dict)
    
    # Renaming columns
    df.columns = df.columns.map("_".join)
    rename_dict = {
        'participant_first': 'participant',
        'probe_first': 'probe',
        'segment_first': 'segment',
        'mind_first': 'mind',
        'mind_category_first': 'mind_category'
    }
    
    # Update rename_dict for mean columns
    for marker in markers:
        rename_dict[f"{marker}_apply_trim_mean"] = f"{marker}_mean"
    
    df = df.rename(columns=rename_dict)

    # Dropping unnecessary columns
    df = df.drop(['probe', 'segment'], axis=1)

    if latex_names:
        # Apply latex naming
        df = correct_name_markers(df)
        df.columns = df.columns.map("$_{".join).map(lambda x: x + '}$').map(lambda x: x.replace('$$', ''))

    # Convert mind category to numeric for analysis
    mind_categories = df['mind_category'].unique()
    mind_category_numeric = {cat: i for i, cat in enumerate(mind_categories)}
    df['mind_numeric'] = df['mind_category'].map(mind_category_numeric)

    # Remove outliers
    columns_to_check = df.drop(['mind_category', 'mind_numeric', 'participant'], axis=1).columns
    # df = replace_outliers_with_participant_mean(df, columns_to_check, z_threshold=3)

    if only_full_participants:
        # Filter participants
        df = df.dropna().groupby('participant').filter(lambda group: filter_participants(group, 'mind_numeric'))

    # Save to CSV if a path is provided
    if results_path:
        df.to_csv(os.path.join(results_path, f'data_{comparison}.csv'))

    return df.dropna()

# Helper functions
from scipy.stats import zscore
def replace_outliers_with_participant_mean(df, columns, participant_column='participant', z_threshold=3):
    df_copy = df.copy()

    # Identify numeric columns
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns

    for col in columns:
        if col in numeric_cols:
            for participant in df_copy[participant_column].unique():
                subset = df_copy[df_copy[participant_column] == participant]
                col_zscore = zscore(subset[col])
                mean_value = np.mean(subset[col][np.abs(col_zscore) < z_threshold])

                # Count the outliers for each participant
                outlier_count = np.sum(np.abs(col_zscore) >= z_threshold)
                total = len(col_zscore)

                # Replace outliers with the mean value for each participant
                subset_indices = subset.index[np.abs(col_zscore) >= z_threshold]
                df_copy.loc[subset_indices, col] = mean_value

                if outlier_count > 0:
                    print(f"Replaced {outlier_count} outliers in column '{col}' out of {total} observations for participant {participant} with the mean value.")
    return df_copy


def filter_participants(group, mind_col_numeric):
    counts = group[mind_col_numeric].value_counts()
    # Check if there is only one level of mind state for the participant
    if len(counts) == 1:
        return False
    return all(count >= 1 for count in counts)

# Define a function to apply the trimmed mean
def apply_trim_mean(group):
    return trim_mean(group, 0.1)

#%%


probe = 'PC'

folds_list=  [4]

# %% [markdown]
# This can only be performed for PC probes  as they are the only ones with On-task reports.


comparisons = ['on-task_vs_mw','on-task_vs_sMW', 'on-task_vs_dMW',  'dMW_vs_sMW']
# comparisons = ['on-task_vs_sMW']
# comparisons = ['dMW_vs_sMW']


# %%
for comparison in comparisons:
    df = preprocess_data(df_markers, markers, probe_type = probe, comparison= comparison, only_full_participants=False, latex_names=False, results_path=None)
    for k in folds_list:
        # Assuming df_mw['participant'] contains the participant IDs
        # Prepare data
        X = df.drop(['mind', 'mind_category', 'mind_numeric', 'participant',], axis=1)
        Z = np.ones((X.shape[0], 1))  # Random effects design matrix
        groups = df['participant']
        y = df['mind_numeric']

        features = df.drop(['mind', 'mind_category', 'mind_numeric', 'participant',], axis=1).columns

        # Construct the file path for the study database
        study_db_path = os.path.join(results_path, 'multivariate_merf_study_final_6.db')

        RM_optimization = RepeatedMeasuresModel.Optimization(
            X, Z, y, groups, k, results_path, 
            database_name = study_db_path, study_name= f'{comparison}_{probe}_K{k}_final_6', n_trials=300, 
            data_augmentation=False,  save_to_df=True
        )

        importances_df = RM_optimization.get_best_model_feature_importances(features)

        importances_df.to_csv(os.path.join(results_path, f'feat_imp_{comparison}_{probe}_K{k}_final_6.csv'))

        RM_optimization.plot_feat_importances(filename = os.path.join(fig_path, f'{comparison}_feat_importance_{probe}_K{k}_final_6.png'), feature_names = features, color = pink, show= False,  save_fig = True)