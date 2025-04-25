#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Full optimization permutation analysis for MVPA models.

This script addresses the reviewer's concerns by:
1. Ensuring permutations occur within-subject, preserving data clustering
2. Running a complete hyperparameter optimization for each permutation
   rather than reusing the original model's hyperparameters
"""

import os
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import trim_mean
import optuna
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from merf import MERF
from joblib import Parallel, delayed
import sys
import time
sys.path.insert(0, '../')

# Path configuration
data_path = "Data/"
results_path = "Results/Multivariate_all_against_all/"
fig_path = results_path + "Figs/"

# Ensure directories exist
os.makedirs(fig_path, exist_ok=True)


def apply_trim_mean(group):
    """Apply a trimmed mean to a group of values."""
    return trim_mean(group, 0.1)


def preprocess_data(df_markers, markers, probe_type, comparison=None, only_full_participants=False, results_path=None):
    """Preprocess the data for analysis based on specified criteria."""
    # Filtering and grouping
    df = df_markers.query(f"probe == '{probe_type}'")
    mind_types = comparison.split('_vs_')
    
    # Adjust mind categories based on comparison type
    if comparison:
        if comparison == 'on-task_vs_mw':
            df['mind_category'] = df['mind'].replace({'dMW': 'mw', 'sMW': 'mw'})
        elif comparison in ['on-task_vs_dMW', 'on-task_vs_sMW', 'dMW_vs_sMW']:
            df = df[df['mind'].isin(mind_types)]
            df['mind_category'] = df['mind']
    else:
        df['mind_category'] = df['mind']

    # Aggregation dictionary
    agg_dict = {k: [apply_trim_mean, 'std'] for k in markers}
    agg_dict.update({k: 'first' for k in df.drop(markers, axis=1).columns})
    df = df.groupby(['segment', 'participant'], as_index=False).agg(agg_dict)
    
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
    mind_category_numeric = {cat: i for i, cat in enumerate(mind_types)}
    df['mind_numeric'] = df['mind_category'].map(mind_category_numeric)

    if only_full_participants:
        # Filter participants
        df = df.dropna().groupby('participant').filter(
            lambda group: filter_participants(group, 'mind_numeric')
        )

    # Save to CSV if a path is provided
    if results_path:
        df.to_csv(os.path.join(results_path, f'data_{comparison}.csv'))

    return df.dropna()


def filter_participants(group, mind_col_numeric):
    """Filter participants to ensure they have at least one sample per mind state."""
    counts = group[mind_col_numeric].value_counts()
    # Check if there is only one level of mind state for the participant
    if len(counts) == 1:
        return False
    return all(count >= 1 for count in counts)


def shuffle_within_participants(y, groups):
    """
    Shuffle target labels within each participant's data.
    This preserves the group structure in the data.
    
    Args:
        y: Target Series
        groups: Group identifiers (participants)
        
    Returns:
        pd.Series: Shuffled version of y that preserves group structure
    """
    y_shuffled = y.copy()
    for participant in groups.unique():
        # Only shuffle within each participant's data
        participant_mask = (groups == participant).values
        if sum(participant_mask) > 1:  # Only shuffle if there's more than one sample
            y_participant = y[participant_mask]
            y_participant_shuffled = np.random.permutation(y_participant)
            y_shuffled[participant_mask] = y_participant_shuffled
    
    return y_shuffled


def optuna_objective(trial, X, y, Z, groups, k=4):
    """
    Objective function for Optuna optimization.
    
    Args:
        trial: Optuna trial
        X, y, Z, groups: Data for model fitting
        k: Number of cross-validation folds
        
    Returns:
        float: Mean AUC score across CV folds
    """
    # RandomForest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 2, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_features = trial.suggest_float('max_features', 0.01, 1)
    criterion = trial.suggest_categorical('criterion', ['absolute_error'])
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
    min_impurity_decrease = trial.suggest_float('min_impurity_decrease', 0.0, 0.1)
    
    # MERF hyperparameters
    gll_early_stop_threshold = trial.suggest_float('gll_early_stop_threshold', 0.005, 0.1, log=True)
    max_iterations = trial.suggest_int('max_iterations', 10, 100)
    
    # Create model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion=criterion,
        bootstrap=bootstrap,
        min_impurity_decrease=min_impurity_decrease,
        random_state=42,
        n_jobs=1  # Use single thread for parallel runs
    )
    
    merf = MERF(
        fixed_effects_model=model,
        gll_early_stop_threshold=gll_early_stop_threshold,
        max_iterations=max_iterations
    )
    
    # Cross-validation
    scores = []
    group_kfold = GroupKFold(n_splits=k)
    
    for train_index, test_index in group_kfold.split(X, y, groups.values):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        clusters_train, clusters_test = groups.iloc[train_index], groups.iloc[test_index]
        
        # Fit and predict
        merf.fit(X_train, Z[train_index], clusters_train, y_train)
        y_pred_proba = expit(merf.predict(X_test, Z[test_index], clusters_test))
        
        # Calculate AUC
        auc = roc_auc_score(y_test, y_pred_proba)
        scores.append(auc)
    
    return np.mean(scores)


def run_permutation_with_full_optimization(
    X, y, Z, groups, k, permutation_id, 
    n_trials=500, 
    dest_db_path=None,
    dest_study_prefix=None
):
    """
    Run a single permutation with full hyperparameter optimization.
    
    Args:
        X, y, Z, groups: Data
        k: Number of CV folds 
        permutation_id: ID for this permutation
        n_trials: Number of optimization trials
        dest_db_path: Path to save optimization study
        dest_study_prefix: Prefix for the study name
        
    Returns:
        dict: Results including best AUC and hyperparameters
    """
    # Shuffle the data within subjects
    y_shuffled = shuffle_within_participants(y, groups)
    
    # Create unique study name for this permutation
    perm_study_name = f"{dest_study_prefix}_perm{permutation_id}"
    
    # Create study
    if dest_db_path is not None:
        storage = f"sqlite:///{dest_db_path}"
        study = optuna.create_study(
            study_name=perm_study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=True
        )
    else:
        study = optuna.create_study(direction="maximize")
    
    # Run optimization
    try:
        study.optimize(
            lambda trial: optuna_objective(trial, X, y_shuffled, Z, groups, k),
            n_trials=n_trials
        )
        
        # Get results
        best_auc = study.best_value
        best_params = study.best_params
        
        return {
            'permutation_id': permutation_id,
            'best_auc': best_auc,
            'best_params': best_params,
            'study_name': perm_study_name if dest_db_path else None
        }
    
    except Exception as e:
        print(f"Error in permutation {permutation_id}: {str(e)}")
        return {
            'permutation_id': permutation_id,
            'best_auc': None,
            'best_params': None,
            'error': str(e)
        }


def run_full_permutation_analysis(
    comparison, 
    probe, 
    k, 
    n_permutations=100, 
    n_trials_per_perm=50,
    source_db='multivariate_merf_study_paper.db',
    source_study_suffix='_paper',
    dest_db='multivariate_merf_full_permutations.db'
):
    """
    Run a complete permutation analysis with full optimization for each permutation.
    
    Args:
        comparison: Type of comparison ('on-task_vs_mw', etc.)
        probe: Probe type ('PC')
        k: Number of CV folds
        n_permutations: Number of permutations
        n_trials_per_perm: Number of trials for each permutation optimization
        source_db: Database filename containing the original study
        source_study_suffix: Suffix for the source study name
        dest_db: Database filename for saving permutation results
        
    Returns:
        DataFrame with permutation results
    """
    start_time = time.time()
    print(f"Starting full permutation analysis for {comparison}, k={k}")
    
    # Load data
    df = pd.read_csv(os.path.join(data_path, 'all_markers.csv'), index_col=0)
    
    # Define markers
    markers = ['wSMI_1', 'wSMI_2', 'wSMI_4', 'wSMI_8', 'p_e_1', 'p_e_2',
        'p_e_4', 'p_e_8', 'k', 'se', 'msf', 'sef90', 'sef95', 'b', 'b_n', 'g',
        'g_n', 't', 't_n', 'd', 'd_n', 'a_n', 'a', 'CNV', 'P1', 'P3a', 'P3b']
    
    # Preprocess data
    df_markers = (df
        .dropna()
        .query("stimuli == 'go'")
        .query("correct == 'correct'")
        .query('prev_trial < 5') 
        .drop(['stimuli', 'correct', 'prev_trial', 'label', 'events', 
               'epoch_type', 'preproc', 'ft', 'ft_n'], axis=1)
        .query("mind in ['on-task','dMW', 'sMW']")
    )
    
    df_processed = preprocess_data(
        df_markers, markers, probe_type=probe, comparison=comparison
    )
    
    # Prepare data
    X = df_processed.drop(['mind', 'mind_category', 'mind_numeric', 'participant'], axis=1)
    Z = np.ones((X.shape[0], 1))  # Random effects design matrix
    groups = df_processed['participant']
    y = df_processed['mind_numeric']
    
    # Load the original study results
    source_db_path = os.path.join(results_path, source_db)
    source_study_name = f'{comparison}_{probe}_K{k}{source_study_suffix}'
    
    try:
        original_study = optuna.load_study(
            study_name=source_study_name, 
            storage=f'sqlite:///{source_db_path}'
        )
        original_auc = original_study.best_value
        print(f"Loaded original study: {source_study_name}")
        print(f"Original model AUC: {original_auc:.4f}")
    except Exception as e:
        print(f"Could not load original study: {e}")
        original_auc = None
    
    # Set up database for permutation results
    dest_db_path = os.path.join(results_path, dest_db)
    dest_study_prefix = f'{comparison}_{probe}_K{k}'
    
    # Run permutations in parallel
    print(f"Running {n_permutations} permutations with {n_trials_per_perm} trials each...")
    permutation_results = Parallel(n_jobs=-1)(
        delayed(run_permutation_with_full_optimization)(
            X, y, Z, groups, k, i, 
            n_trials=n_trials_per_perm,
            dest_db_path=dest_db_path,
            dest_study_prefix=dest_study_prefix
        )
        for i in range(n_permutations)
    )
    
    # Compile results
    permutation_df = pd.DataFrame(permutation_results)
    
    # Calculate p-value if original study was loaded
    if original_auc is not None:
        perm_scores = permutation_df['best_auc'].dropna()
        p_value = np.mean(perm_scores >= original_auc)
        print(f"P-value: {p_value:.4f}")
        
        # Add original performance to dataframe
        permutation_df = pd.concat([
            pd.DataFrame([{
                'permutation_id': 'original',
                'best_auc': original_auc,
                'best_params': str(original_study.best_params),
                'study_name': source_study_name
            }]),
            permutation_df
        ])
    else:
        p_value = None
    
    # Save results
    perm_mean = permutation_df['best_auc'].mean()
    perm_std = permutation_df['best_auc'].std()
    
    # Create visualization
    try:
        import matplotlib.pyplot as plt
        perm_scores = permutation_df.loc[permutation_df['permutation_id'] != 'original', 'best_auc'].dropna()
        
        plt.figure(figsize=(10, 6))
        plt.hist(perm_scores, bins=30, alpha=0.7, label='Permutation scores')
        
        if original_auc is not None:
            plt.axvline(original_auc, color='red', linestyle='dashed', 
                        linewidth=2, label=f'Original AUC: {original_auc:.4f}')
            plt.axvline(perm_mean, color='green', linestyle='dotted', 
                        linewidth=2, label=f'Permutation mean: {perm_mean:.4f}')
            plt.title(f'Full Optimization Permutation - {comparison} (p={p_value:.4f})')
        else:
            plt.axvline(perm_mean, color='green', linestyle='dotted', 
                        linewidth=2, label=f'Permutation mean: {perm_mean:.4f}')
            plt.title(f'Full Optimization Permutation - {comparison}')
            
        plt.xlabel('AUC Score')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Save the histogram
        hist_path = os.path.join(fig_path, f'full_perm_dist_{comparison}_{probe}_K{k}.png')
        plt.savefig(hist_path)
        print(f"Saved permutation histogram to {hist_path}")
        plt.close()
    except Exception as e:
        print(f"Could not create histogram: {e}")
    
    # Save results to CSV
    csv_path = os.path.join(results_path, f'full_permutations_{comparison}_{probe}_K{k}.csv')
    permutation_df.to_csv(csv_path, index=False)
    
    # Save summary stats
    summary = {
        'comparison': comparison,
        'probe': probe,
        'k_folds': k,
        'original_auc': original_auc,
        'perm_mean': perm_mean,
        'perm_std': perm_std,
        'p_value': p_value,
        'n_permutations': n_permutations,
        'n_trials_per_perm': n_trials_per_perm,
        'runtime_minutes': (time.time() - start_time) / 60
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(results_path, f'full_perm_summary_{comparison}_{probe}_K{k}.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print(f"Completed in {(time.time() - start_time)/60:.1f} minutes")
    print(f"Summary: Original AUC={original_auc:.4f}, Permutation mean={perm_mean:.4f}, p-value={p_value:.4f}")
    
    return permutation_df


if __name__ == "__main__":
    probe = 'PC'
    folds_list = [4]
    comparisons = ['on-task_vs_mw', 'on-task_vs_dMW', 'on-task_vs_sMW', 'dMW_vs_sMW']
    
    # Example usage - uncomment to run
    for comparison in comparisons:
        for k in folds_list:
            print(f"Running full optimization permutation analysis for {comparison}, k={k}")
            run_full_permutation_analysis(
                comparison=comparison,
                probe=probe, 
                k=k, 
                n_permutations=500,  # This is computationally expensive
                n_trials_per_perm=50,  # Reduced number of trials per permutation
                source_db='multivariate_merf_study_paper.db',
                source_study_suffix='_paper',
                dest_db='multivariate_merf_full_permutations.db'
            ) 