# %%
import sys

from scipy.special import expit 


import os
import numpy as np
import pandas as pd 
from tqdm.notebook import tqdm


import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as pgo
import plotly.offline as pyo
import plotly.io as pio
from plotly.subplots import make_subplots

pyo.init_notebook_mode(connected = True)

from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor

from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from pymer4.models import Lmer
from merf import MERF

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

import optuna

# %%
# plotting parameters
grey = "#21201F"
green = "#9AC529"
lblue = "#42B9B2"
pink = "#DE237B"
orange = "#F38A31"

nt_colors = [green, lblue, pink, orange]

plt.style.use("ggplot")
fig_width = 2  # width in inches
fig_height = 8  # height in inches
fig_size = [fig_width, fig_height]
plt.rcParams["figure.figsize"] = fig_size
plt.rcParams["figure.autolayout"] = True

sns.set(
    style="white",
    context="notebook",
    font_scale=1.5,
    rc={
        "axes.labelcolor": grey,
        "text.color": grey,
        "axes.edgecolor": grey,
        "xtick.color": grey,
        "ytick.color": grey,
        'figure.figsize': fig_size
    },
)

sns.set_palette(sns.color_palette(nt_colors))

# %% [markdown]
# # Load Data

# %%
data_path = "./Data/"
results_path = "./Results/"
fig_path = "./Results/Figs/"

df = pd.read_csv(os.path.join(data_path, 'all_markers.csv'), index_col = 0)

# %%
#markers names
all_participants = ['VP07','VP08','VP09', 'VP10','VP11','VP12','VP13','VP14','VP18','VP19','VP20','VP22','VP23','VP24','VP25','VP26','VP27','VP28','VP29','VP30','VP31','VP32','VP33','VP35','VP36','VP37']
#selection of good participants. Not used.
good_participants = all_participants[1:2] +  all_participants[6:10] +  all_participants[12:15]  + all_participants[18:23] + [all_participants[25]]
len(good_participants)

# %%
markers = ['wSMI_1', 'wSMI_2', 'wSMI_4', 'wSMI_8', 'p_e_1', 'p_e_2',
       'p_e_4', 'p_e_8', 'k', 'se','msf', 'sef90', 'sef95', 'b', 'b_n', 'g',
       'g_n', 't', 't_n', 'd', 'd_n', 'a_n', 'a', 'CNV', 'P1', 'P3a', 'P3b',]
#           'ft', 'ft_n']
erps =['CNV', 'P1', 'P3a', 'P3b']
# erps = [r'$CNV$', r'$P1$', r'$P3a$',r'$P3b$']

# markers =  [r'$\delta$',r'$|\delta|$',r'$\theta$', r'$|\theta|$',r'$\alpha$', r'$|\alpha|$',r'$\beta$', r'$|\beta|$',r'$\gamma$', r'$|\gamma|$',
#             r'$PE\gamma$',r'$PE\beta$',r'$PE\alpha$',r'$PE\theta$',
#             r'$wSMI\gamma$',r'$wSMI\beta$',r'$wSMI\alpha$',r'$wSMI\theta$', 
#             r'$K$',r'$SE$',r'$MSF$', r'$SEF90$', r'$SEF95$', 
#             r'$CNV$', r'$P1$', r'$P3a$',r'$P3b$'
#            ]


# df_subtracted = df.query("preproc == 'subtracted'").drop(columns = erps+['preproc'])
# df_erp = df.query("preproc == 'erp'").drop(columns = np.setdiff1d(markers,erps).tolist()+['preproc'])

# df_markers = df_subtracted.merge(df_erp, 'inner', on =np.setdiff1d(df_subtracted.columns, markers).tolist() )

df_markers = (df
              .query("stimuli == 'go'")
              .query("correct == 'correct'")
              .query('prev_trial < 5')
              .drop(['stimuli', 'correct', 'prev_trial', 'label', 'events',  'epoch_type', 'preproc', 'ft', 'ft_n'], axis = 1)
              .query("mind in ['on-task','dMW', 'sMW']")
              .groupby(['segment', 'participant']).filter(lambda x: len(x) > 1)
             )

df_markers['segment'] = df_markers['segment'].str.replace('s', '').astype(int)

# %%
# display(df_mind.groupby(['participant','mind2', ]).count(),
# # df_mw.groupby(['participant','mind', ]).count()
# )

# %% [markdown]
# # By Segment Univariate analyses

# %% [markdown]
# ## On-task Vs Mind- Wandering
# This can only be performed for PC probes  as they are the only ones with On-task reports.

# %%
agg_dict = {k:['mean', 'std'] for k in markers }
agg_dict.update({k:'first' for k in df_markers.drop(markers, axis=1).columns})

df_mind = (
    df_markers
    .query("probe == 'PC'")
    .groupby(['segment', 'participant'], as_index = False).agg(agg_dict)
    .assign(
    mind2 = lambda df: np.where(df.mind == 'on-task', 'on-task', 'mw'))
)


############################################################
################ Use normal names################
############################################################

df_mind.columns = df_mind.columns.map("_".join)

df_mind  = (df_mind
            .rename(columns = {'participant_first':'participant', 'probe_first':'probe', 'mind_first':'mind', 'segment_first':'segment', 'mind2_':'mind2'})
            # .query("mind != 'dMW'") #if you want to test against just one of the mw            
            .drop([ 'probe', 'mind',], axis = 1) 
           )

############################################################
################ Use latex command for nmaes################
############################################################

##it slow downs the computer, just for final figures.

# df_mind = correct_name_markers(df_mind)

# df_mind.columns = df_mind.columns.map("$_{".join).map(lambda x: x + '}$').map(lambda x: x.replace('$$', ''))

# df_mind  = (df_mind
#             .rename(columns = {'participant$_{first}$':'participant', 'probe$_{first}$':'probe', 'mind$_{first}$':'mind', 'segment$_{first}$':'segment', 'mind2$_{}$':'mind2'})
# #             .query("mind != 'dMW'") #if you want to test against just one of the mw            
#             .drop(['probe', 'mind', 'segment'], axis = 1) 
        #    )
        
df_mind['mind2_numeric'] = (df_mind['mind2'] == 'mw').astype(int)

# def filter_participants(group):
#     counts = group['mind2'].value_counts()
#     # Check if there is only one level of 'mind2' for the participant
#     if len(counts) == 1:
#         return False
#     return all(count >= 2 for count in counts)

# df_mind = df_mind.groupby('participant').filter(filter_participants)


from scipy.stats import zscore
def replace_outliers_with_participant_mean(df, columns, participant_column='participant', z_threshold=3):
    df_copy = df.copy()
    
    for col in columns:
        for participant in df[participant_column].unique():
            subset = df[df[participant_column] == participant]
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
# Columns to remove outliers from
columns_to_check = df_mind.drop(['mind2', 'mind2_numeric', 'participant', 'segment'], axis = 1).columns

# Remove outliers
df_mind_filtered = replace_outliers_with_participant_mean(df_mind, columns_to_check, z_threshold=3)

df_mind = df_mind_filtered

# %%
def objective(trial, df_mind):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 50, 200)
    max_depth = trial.suggest_int("max_depth", 5, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 4)
    max_features = trial.suggest_uniform("max_features", 0.5, 1.0)

    # Initialize results DataFrame for this trial
    results_df_trial = pd.DataFrame(columns=['Marker', 'AUC_mean', 'AUC_std', 'AUC_sem', 'AUC_range'])
    total_auc = 0

    # Loop through each marker
    for marker in tqdm(df_mind.drop(['mind2', 'mind2_numeric', 'participant', 'segment'], axis=1).columns, desc="Markers"):
        df_marker = df_mind[['mind2', 'mind2_numeric', 'participant', 'segment', marker]]

        # Prepare data
        X = df_marker[marker]
        Z = np.ones((X.shape[0], 1))  # Random effects design matrix
        clusters = df_marker['participant']
        y = df_marker['mind2_numeric']
    
        n_splits = 5
        group_kfold = GroupKFold(n_splits=n_splits)

        # Placeholder for AUC scores and optimal cutoffs
        auc_scores = []
        optimal_cutoffs = []

        # Assuming df_marker['participant'] contains the participant IDs
        groups = df_marker['participant'].values

        # Initialize MERF with suggested parameters
        merf = MERF(RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1
        ))

        for train_index, test_index in group_kfold.split(X, y, groups):
            df_train = df_marker.iloc[train_index]
            df_train[marker] = StandardScaler().fit_transform(df_train[marker].values.reshape(-1, 1))
            df_test = df_marker.iloc[test_index]
            df_test[marker] = StandardScaler().fit_transform(df_test[marker].values.reshape(-1, 1))

            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            clusters_train, clusters_test = clusters.iloc[train_index], clusters.iloc[test_index]

            merf.fit(df_train[marker].values.reshape(-1, 1), Z[train_index], clusters_train, y_train)
            
            y_pred = merf.predict(df_test[marker].values.reshape(-1, 1), Z[test_index], clusters_test)

            # Convert continuous outputs to probabilities
            y_pred_proba = expit(y_pred)
            
            # Find optimal cutoff point based on F1 score
            thresholds = np.linspace(0, 1, 100)
            f1_scores = [f1_score(y_test, (y_pred_proba > t).astype(int)) for t in thresholds]
            optimal_threshold = thresholds[np.argmax(f1_scores)]
            optimal_cutoffs.append(optimal_threshold)
            
            # Compute AUC for probabilities
            auc = roc_auc_score(y_test, y_pred_proba)
            auc_scores.append(auc)

        # Compute AUC statistics for the marker
        auc_mean = np.mean(auc_scores)
        auc_std = np.std(auc_scores)
        auc_sem = auc_std / np.sqrt(len(auc_scores))
        auc_range = np.ptp(auc_scores)

        # Append results to trial's DataFrame
        results_df_trial = results_df_trial.append({
            'Marker': marker,
            'AUC_mean': auc_mean,
            'AUC_std': auc_std,
            'AUC_sem': auc_sem,
            'AUC_range': auc_range
        }, ignore_index=True)

        # Update total AUC
        total_auc += auc_mean

    # Attach the trial's DataFrame to the trial as a user attribute
    trial.set_user_attr("results_df", results_df_trial)

    # Calculate and return the average AUC
    return total_auc / len(df_mind.drop(['mind2', 'mind2_numeric', 'participant', 'segment'], axis=1).columns)

# Run Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(lambda trial: objective(trial, df_mind), n_trials=50)

# Retrieve the best trial's results DataFrame
best_trial = study.best_trial
print("Best trial:", best_trial.params)
best_results_df = best_trial.user_attrs["results_df"]

# Save the best trial's results to a CSV file
best_results_csv_path = os.path.join(results_path, 'univariate_merf_mind_best.csv')
best_results_df.to_csv(best_results_csv_path)

# Save the best trial's parameters to a file
best_params_path = os.path.join(results_path, 'univariate_merf_mind_best_params.txt')
with open(best_params_path, 'w') as file:
    for key, value in best_trial.params.items():
        file.write(f'{key}: {value}\n')

print(f"Results of the best trial saved in: {best_results_csv_path}")

# %%
mind_merf = pd.read_csv(os.path.join(results_path, 'univariate_merf_mind_best.csv'))

# segment_mw_roc = segment_mw_roc.sort_values(by = 'AUC', ascending = False).head(10).append(segment_mw_roc.sort_values(by = 'AUC', ascending = False).tail(10))

fig = px.scatter(mind_merf.sort_values(by = 'AUC_mean'),x = 'AUC_mean', y = 'Marker', template = "plotly_white", 
                
#                  color = 'significant',
                 color_discrete_sequence = [pink, green,orange, pink], 
                 
                 category_orders = {'significant': ['p > 0.05','p < 0.05 uncorrected', 'p < 0.05 FDR corrected']},
                 labels = {'AUC': 'sTUT>dTUT              sTUT<dTUT', }
                )
fig.add_vline(x=0.5, line_width=3, line_dash="dash", line_color="grey")
fig.update_traces(marker=dict(size = 13))

fig.update_layout(
    width=850,
    height=1300,
#     autosize = True, 
    template = 'plotly_white',
        font=dict(
        family="Times new roman",
        size=20,
        color="black"
    ),
    xaxis = dict(
            visible=True,
            range = [0.45,0.75], 
            tickfont = {"size": 20},
        ),
    yaxis = dict(
        tickfont = {"size": 20},
        autorange = False,    
        automargin = True,
        range = [-1,len(mind_merf)],
        dtick = 1
        ),
    showlegend=True, 

)

# fig.show()

fig.write_image(os.path.join(fig_path,'univariate_merf_mind_best.png'))
fig.write_image(os.path.join(fig_path,'univariate_merf_mind_best.pdf'))
fig.write_html(os.path.join(fig_path,'univariate_merf_mind_best.html'))

# %% [markdown]
# ## dMW Vs sMW
# This will be only performed in SC as they have more trials

# %%
agg_dict = {k:['mean', 'std'] for k in markers }
agg_dict.update({k:'first' for k in df_markers.drop(markers, axis=1).columns})

df_mw = (
    df_markers
    .query("probe == 'SC'")
    .query("mind != 'on-task'")
    .groupby(['segment', 'participant'], as_index = False).agg(agg_dict)
)

############################################################
################ Use normal names################
############################################################
df_mw.columns = df_mw.columns.map("_".join)

df_mw  = (df_mw
            .rename(columns = {'participant_first':'participant', 'probe_first':'probe', 'mind_first':'mind', 'segment_first':'segment'})
            .drop([ 'probe',], axis = 1)
            )


############################################################
################ Use latex command for nmaes################
############################################################

# df_mw = correct_name_markers(df_mw)

# df_mw.columns = df_mw.columns.map("$_{".join).map(lambda x: x + '}$').map(lambda x: x.replace('$$', ''))

# df_mw  = (df_mw
#             .rename(columns = {'participant$_{first}$':'participant', 'probe$_{first}$':'probe', 'mind$_{first}$':'mind', 'segment$_{first}$':'segment', 'mind$_{}$':'mind'})
# #             .query("mind != 'dMW'") #if you want to test against just one of the mw   
#             .drop(['participant', 'probe',  'segment'], axis = 1)

#            )


df_mw['mind_numeric'] = (df_mw['mind'] == 'sMW').astype(int)

df_mw.to_csv(os.path.join(results_path,'data_mw.csv'))

from scipy.stats import zscore
def replace_outliers_with_participant_mean(df, columns, participant_column='participant', z_threshold=3):
    for col in columns:
        for participant in df[participant_column].unique():
            subset = df[df[participant_column] == participant]
            col_zscore = zscore(subset[col])
            mean_value = np.mean(subset[col][np.abs(col_zscore) < z_threshold])
            
            # Count the outliers for each participant
            outlier_count = np.sum(np.abs(col_zscore) >= z_threshold)
            
            total = len(col_zscore)
            
            
            # Replace outliers with the mean value for each participant
            df.loc[(np.abs(col_zscore) >= z_threshold) & (df[participant_column] == participant), col] = mean_value
            
            # Print the number of outliers replaced for each participant and column
            if outlier_count > 0:
                print(f"Replaced {outlier_count} outliers in column '{col}' out of {total} observations for participant {participant} with the mean value.")
                
    return df

# Columns to remove outliers from
columns_to_check = df_mw.drop(['mind', 'mind_numeric', 'participant', 'segment'], axis = 1).columns

# Remove outliers
df_mw_filtered = replace_outliers_with_participant_mean(df_mw, columns_to_check, z_threshold=3)

df_mw = df_mw_filtered


# %%
def objective(trial, df_mw):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 50, 200)
    max_depth = trial.suggest_int("max_depth", 5, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 4)
    max_features = trial.suggest_uniform("max_features", 0.5, 1.0)

    # Initialize results DataFrame for this trial
    results_df_trial = pd.DataFrame(columns=['Marker', 'AUC_mean', 'AUC_std', 'AUC_sem', 'AUC_range'])
    total_auc = 0

    # Loop through each marker
    for marker in tqdm(df_mw.drop(['mind', 'mind_numeric', 'participant', 'segment'], axis=1).columns, desc="Markers"):
        df_marker = df_mw[['mind', 'mind_numeric', 'participant', 'segment', marker]]

        # Prepare data
        X = df_marker[marker]
        Z = np.ones((X.shape[0], 1))  # Random effects design matrix
        clusters = df_marker['participant']
        y = df_marker['mind_numeric']
    
        n_splits = 5
        group_kfold = GroupKFold(n_splits=n_splits)

        # Placeholder for AUC scores and optimal cutoffs
        auc_scores = []
        optimal_cutoffs = []

        # Assuming df_marker['participant'] contains the participant IDs
        groups = df_marker['participant'].values

        # Initialize MERF with suggested parameters
        merf = MERF(RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1
        ))

        for train_index, test_index in group_kfold.split(X, y, groups):
            df_train = df_marker.iloc[train_index]
            df_train[marker] = StandardScaler().fit_transform(df_train[marker].values.reshape(-1, 1))
            df_test = df_marker.iloc[test_index]
            df_test[marker] = StandardScaler().fit_transform(df_test[marker].values.reshape(-1, 1))

            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            clusters_train, clusters_test = clusters.iloc[train_index], clusters.iloc[test_index]

            merf.fit(df_train[marker].values.reshape(-1, 1), Z[train_index], clusters_train, y_train)
            
            y_pred = merf.predict(df_test[marker].values.reshape(-1, 1), Z[test_index], clusters_test)

            # Convert continuous outputs to probabilities
            y_pred_proba = expit(y_pred)
            
            # Find optimal cutoff point based on F1 score
            thresholds = np.linspace(0, 1, 100)
            f1_scores = [f1_score(y_test, (y_pred_proba > t).astype(int)) for t in thresholds]
            optimal_threshold = thresholds[np.argmax(f1_scores)]
            optimal_cutoffs.append(optimal_threshold)
            
            # Compute AUC for probabilities
            auc = roc_auc_score(y_test, y_pred_proba)
            auc_scores.append(auc)

        # Compute AUC statistics for the marker
        auc_mean = np.mean(auc_scores)
        auc_std = np.std(auc_scores)
        auc_sem = auc_std / np.sqrt(len(auc_scores))
        auc_range = np.ptp(auc_scores)

        # Append results to trial's DataFrame
        results_df_trial = results_df_trial.append({
            'Marker': marker,
            'AUC_mean': auc_mean,
            'AUC_std': auc_std,
            'AUC_sem': auc_sem,
            'AUC_range': auc_range
        }, ignore_index=True)

        # Update total AUC
        total_auc += auc_mean

    # Attach the trial's DataFrame to the trial as a user attribute
    trial.set_user_attr("results_df", results_df_trial)

    # Calculate and return the average AUC
    return total_auc / len(df_mw.drop(['mind', 'mind_numeric', 'participant', 'segment'], axis=1).columns)

# Run Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(lambda trial: objective(trial, df_mw), n_trials=50)

# Retrieve the best trial's results DataFrame
best_trial = study.best_trial
print("Best trial:", best_trial.params)
best_results_df = best_trial.user_attrs["results_df"]

# Save the best trial's results to a CSV file
best_results_csv_path = os.path.join(results_path, 'univariate_merf_mw_best.csv')
best_results_df.to_csv(best_results_csv_path)

# Save the best trial's parameters to a file
best_params_path = os.path.join(results_path, 'univariate_merf_mw_best_params.txt')
with open(best_params_path, 'w') as file:
    for key, value in best_trial.params.items():
        file.write(f'{key}: {value}\n')

print(f"Results of the best trial saved in: {best_results_csv_path}")

# %%
mw_merf = pd.read_csv(os.path.join(results_path, 'univariate_merf_mw_best.csv'))

# segment_mw_roc = segment_mw_roc.sort_values(by = 'AUC', ascending = False).head(10).append(segment_mw_roc.sort_values(by = 'AUC', ascending = False).tail(10))

fig = px.scatter(mw_merf.sort_values(by = 'AUC_mean'),x = 'AUC_mean', y = 'Marker', template = "plotly_white", 
                
#                  color = 'significant',
                 color_discrete_sequence = [lblue, green,orange, pink], 
                 
                 category_orders = {'significant': ['p > 0.05','p < 0.05 uncorrected', 'p < 0.05 FDR corrected']},
                 labels = {'AUC': 'sTUT>dTUT              sTUT<dTUT', }
                )
fig.add_vline(x=0.5, line_width=3, line_dash="dash", line_color="grey")
fig.update_traces(marker=dict(size = 13))

fig.update_layout(
    width=850,
    height=1300,
#     autosize = True, 
    template = 'plotly_white',
        font=dict(
        family="Times new roman",
        size=20,
        color="black"
    ),
    xaxis = dict(
            visible=True,
            range = [0.45,0.75], 
            tickfont = {"size": 20},
        ),
    yaxis = dict(
        tickfont = {"size": 20},
        autorange = False,    
        automargin = True,
        range = [-1,len(mw_merf)],
        dtick = 1
        ),
    showlegend=True, 

)

# fig.show()

fig.write_image(os.path.join(fig_path,'univariate_merf_mw_best.png'))
fig.write_image(os.path.join(fig_path,'univariate_merf_mw_best.pdf'))
fig.write_html(os.path.join(fig_path,'univariate_merf_mw_best.html'))


