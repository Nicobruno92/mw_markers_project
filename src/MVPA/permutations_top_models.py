# %%
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import expit 

from scipy.stats import trim_mean

import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.io as pio
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold

from merf import MERF

import optuna

from joblib import Parallel, delayed

import sys
sys.path.insert(0, '../')


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
              .query('prev_trial < 5') # only last 5 trials before each probe. 
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

    # Convert mind category to numeric for analysis
    mind_categories = df['mind_category'].unique()
    mind_category_numeric = {cat: i for i, cat in enumerate(mind_categories)}
    df['mind_numeric'] = df['mind_category'].map(mind_category_numeric)

    if only_full_participants:
        # Filter participants
        df = df.dropna().groupby('participant').filter(lambda group: filter_participants(group, 'mind_numeric'))

    # Save to CSV if a path is provided
    if results_path:
        df.to_csv(os.path.join(results_path, f'data_{comparison}.csv'))

    return df.dropna()


def filter_participants(group, mind_col_numeric):
    counts = group[mind_col_numeric].value_counts()
    # Check if there is only one level of mind state for the participant
    if len(counts) == 1:
        return False
    return all(count >= 1 for count in counts)

# Define a function to apply the trimmed mean
def apply_trim_mean(group):
    return trim_mean(group, 0.1)

def perform_permutations(model, X, y, Z, groups, k=4):
    scores = []
    optimal_cutoffs = []
    
    group_kfold = GroupKFold(n_splits=k)

    y_shuffled = np.random.permutation(y)
    for train_index, test_index in group_kfold.split(X, y_shuffled, groups.values):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_shuffled[train_index], y_shuffled[test_index]
        clusters_train, clusters_test = groups.iloc[train_index], groups.iloc[test_index]

        merf.fit(X_train, Z[train_index], clusters_train, y_train)
        y_pred_proba = expit(merf.predict(X_test, Z[test_index], clusters_test))

        # Evaluate model
        thresholds = np.linspace(0, 1, 100)
        f1_scores = [f1_score(y_test, (y_pred_proba > t).astype(int)) for t in thresholds]
        optimal_threshold = thresholds[np.argmax(f1_scores)]

        # Compute AUC for probabilities
        auc = roc_auc_score(y_test, y_pred_proba)
        scores.append(auc)
        optimal_cutoffs.append(optimal_threshold)
    
    return np.mean(scores)
#%%


probe = 'PC'

folds_list=  [4]

# This can only be performed for PC probes  as they are the only ones with On-task reports.


comparisons = ['on-task_vs_mw','on-task_vs_sMW', 'on-task_vs_dMW',  'dMW_vs_sMW']
# comparisons = ['on-task_vs_sMW']
# comparisons = ['dMW_vs_sMW']

n_permutations = 500

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
        study_db_path = os.path.join(results_path, 'multivariate_merf_study_final.db')
        
                # Load the Optuna study
        study = optuna.load_study(study_name= f'{comparison}_{probe}_K{k}_trim', storage=f'sqlite:///{study_db_path}')

        # print([trial.values for trial in study.trials])

        # Retrieve the top 10 best trials
        # Filter out trials with None values and then retrieve the top 10 best trials
        top_trials = sorted([trial for trial in study.trials if trial.value is not None], 
                            key=lambda trial: trial.value, reverse=True)[:10]

        results = []
        for trial in top_trials:
            original_auc = trial.value
            best_params = trial.params
            model = RandomForestRegressor(
                    n_estimators= best_params["n_estimators"],
                    max_depth= best_params["max_depth"],
                    min_samples_split= best_params["min_samples_split"],
                    min_samples_leaf= best_params["min_samples_leaf"],
                    max_features= best_params["max_features"],
                    criterion= best_params["criterion"] ,
                    bootstrap= best_params["bootstrap"],
                    min_impurity_decrease= best_params["min_impurity_decrease"],
                    random_state=42,
                    n_jobs=-1
                )
            merf = MERF(fixed_effects_model=model, gll_early_stop_threshold=best_params["gll_early_stop_threshold"], max_iterations=best_params["max_iterations"])
            
            permutation_scores = Parallel(n_jobs=-1)(
                delayed(perform_permutations)(merf, X, y, Z, groups)
                for _ in range(n_permutations)
            )

            auc_pvalue = np.mean([score >= original_auc for score in permutation_scores])

            results.append({
                'trial_number': trial.number,
                'original_auc': original_auc,
                'perm_mean_auc': np.mean(permutation_scores),
                'perm_std_auc': np.std(permutation_scores),
                'p_value': auc_pvalue, 
                'permutations_auc': permutation_scores,
            })

        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(results_path, f'permutations_{comparison}_{probe}_K{k}_trim.csv'))