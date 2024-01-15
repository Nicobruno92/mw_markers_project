# %%
import os
import numpy as np
import pandas as pd 
from tqdm import tqdm


import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as pgo
import plotly.offline as pyo
import plotly.io as pio
from plotly.subplots import make_subplots

from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from pymer4.models import Lmer

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from joblib import Parallel, delayed

np.random.seed(42)

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
# metric = 'kappa'
# metric = 'f1'
# metric = 'auc'
# Execute the loop in parallel
n_permutations = 5000  # You can adjust this

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

def perform_permutation(formula, df_train, df_test, y_test, original_labels, label,):
    # Make a copy of the original labels
    y_perm = np.random.permutation(original_labels)
    df_train[label] = y_perm
    model_perm = Lmer(formula, data=df_train, family="binomial")
    model_perm.fit(verbose=False, summary=False)

    # Make predictions and compute AUC for permuted labels
    predicted_probabilities_perm = model_perm.predict(df_test, use_rfx=True, verify_predictions=False)
    predicted_labels = [1 if prob > 0.5 else 0 for prob in predicted_probabilities_perm]
    
    
    perm_auc = roc_auc_score(y_test, predicted_probabilities_perm)
    perm_f1 = f1_score(y_test, predicted_labels)
    perm_kappa = cohen_kappa_score(y_test, predicted_labels)
    
    return perm_auc, perm_f1, perm_kappa


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
#             .query("mind != 'dMW'") #if you want to test against just one of the mw            
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
# Columns to remove outliers from
columns_to_check = df_mind.drop(['mind2', 'mind2_numeric', 'participant', 'segment'], axis = 1).columns

def filter_participants(group):
    counts = group['mind2_numeric'].value_counts()
    # Check if there is only one level of 'mind2' for the participant
    if len(counts) == 1:
        return False
    return all(count >= 1 for count in counts)
df_mind = df_mind.groupby('participant').filter(filter_participants)

# Remove outliers
df_mind_filtered = replace_outliers_with_participant_mean(df_mind, columns_to_check, z_threshold=3)

# df_mind = df_mind_filtered
# %% [markdown]
# ## dMW Vs sMW
# This will be only performed in SC as they have more trials

# %%
agg_dict = {k:['mean', 'std'] for k in markers }
agg_dict.update({k:'first' for k in df_markers.drop(markers, axis=1).columns})

df_mw = (
    df_markers
    .query("probe == 'PC'")
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

# Columns to remove outliers from
columns_to_check = df_mw.drop(['mind', 'mind_numeric', 'participant', 'segment'], axis = 1).columns

def filter_participants(group):
    counts = group['mind_numeric'].value_counts()
    # Check if there is only one level of 'mind2' for the participant
    if len(counts) == 1:
        return False
    return all(count >= 1 for count in counts)
df_mw = df_mw.groupby('participant').filter(filter_participants)
# Remove outliers
df_mw_filtered = replace_outliers_with_participant_mean(df_mw, columns_to_check, z_threshold=3)

# df_mw = df_mw_filtered

# %%
folds = [4,5,6,]
for k in folds:
    n_splits = k
    
    results_df = pd.DataFrame(columns=['Marker', 'Log-Likelihood', 'AIC','Estimate', 'P_val', 
                                    'score_mean', 'score_std', 'score_sem', 'score_range', 'perm_score','perms_score_std','perms_score_all'])


    for marker in tqdm(df_mind.drop(['mind2', 'mind2_numeric', 'participant', 'segment'], axis = 1).columns, desc="Markers"):
        formula = f"mind2_numeric ~ {marker} + (1|participant)"
        
        other_markers = np.setdiff1d(df_mind.drop(['mind2', 'mind2_numeric', 'participant', 'segment'], axis = 1).columns, marker)
        df_marker = df_mind.drop(other_markers, axis = 1).reset_index()
        
        # Fitting the LMER model
        base_model = Lmer(formula, data=df_marker, family="binomial")
        base_model.fit(verbose = False, summary = False)
        
        # Stratified KFold for ROC AUC
        # skf = StratifiedKFold(n_splits=5)
        X = df_marker[marker].values.reshape(-1, 1)
        y = df_marker['mind2_numeric']
        
        auc_scores = []
        f1_scores = []
        kappa_scores = []
        
        # Initialize the lists outside the loop
        perm_auc, perm_f1, perm_kappa = [], [], []
        
        # n_splits = 5
        group_kfold = GroupKFold(n_splits=n_splits)

        # Assuming df_mw['participant'] contains the participant IDs
        groups = df_marker['participant'].values

        for train_index, test_index in group_kfold.split(X, y, groups):
            df_train = df_marker.iloc[train_index]
            # df_train[marker] = StandardScaler().fit_transform(df_train[marker].values.reshape(-1, 1))
            df_test = df_marker.iloc[test_index]
            # df_test[marker] = StandardScaler().fit_transform(df_test[marker].values.reshape(-1, 1))
            y_test = y[test_index]
            # Save original labels
            original_labels = df_train['mind2_numeric'].copy()
            
            model = Lmer(formula, data=df_train, family="binomial")
            model.fit(verbose = False, summary = False)
            

            predicted_probabilities = model.predict(df_test, use_rfx=True, verify_predictions=False)
            predicted_labels = [1 if prob > 0.5 else 0 for prob in predicted_probabilities]

            auc = roc_auc_score(y_test, predicted_probabilities)
            auc_scores.append(auc)
            f1 = f1_score(y_test, predicted_labels)
            f1_scores.append(f1)
            kappa = cohen_kappa_score(y_test, predicted_labels)
            kappa_scores.append(kappa)
            
            # Permutation test
            perm_scores = Parallel(n_jobs=-1)(
                delayed(perform_permutation)(formula, df_train.copy(), df_test, y_test, original_labels, label='mind2_numeric')
                for _ in range(n_permutations)
            )

            # Append the mean of the scores
            perm_auc.append([score[0] for score in perm_scores])
            perm_f1.append([score[1] for score in perm_scores])
            perm_kappa.append([score[2] for score in perm_scores])
            

        # Convert 2D list to DataFrame
        df_perm_auc = pd.DataFrame(perm_auc).T
        df_perm_f1 = pd.DataFrame(perm_f1).T
        df_perm_kappa = pd.DataFrame(perm_kappa).T
        

        # Average across folds for each permutation run
        df_perm_auc['Mean_score'] = df_perm_auc.mean(axis=1)
        df_perm_f1['Mean_score'] = df_perm_f1.mean(axis=1)
        df_perm_kappa['Mean_score'] = df_perm_kappa.mean(axis=1)
            
        
        # Compute AUC statistics
        auc_mean = np.mean(auc_scores)
        auc_std = np.std(auc_scores)
        auc_pvalue = np.mean(df_perm_auc['Mean_score'] >= auc_mean)
        
        # Compute F1 statistics
        f1_mean = np.mean(f1_scores)
        f1_std = np.std(f1_scores)
        f1_pvalue = np.mean(df_perm_f1['Mean_score'] >= f1_mean)
        
        # Compute F1 statistics
        kappa_mean = np.mean(kappa_scores)
        kappa_std = np.std(kappa_scores)
        kappa_pvalue = np.mean(df_perm_kappa['Mean_score'] >= kappa_mean)
        
        
        # Save to DataFrame
        results_df = results_df.append({
            'Marker': marker,
            'Log-Likelihood': base_model.logLike,
            'AIC': base_model.AIC,
            'Glmm_pval': base_model.coefs['P-val'][0],
            'Estimate': base_model.coefs['Estimate'][0],
            'AUC_P_val': auc_pvalue,
            'AUC_mean': auc_mean,
            'AUC_std': auc_std,
            'F1_P_val': f1_pvalue,
            'F1_mean': f1_mean,
            'F1_std': f1_std,
            'Kappa_P_val': kappa_pvalue,
            'Kappa_mean': kappa_mean,
            'Kappa_std': kappa_std,
            'perm_auc_mean': df_perm_auc['Mean_score'].mean(),
            'perm_auc_all': df_perm_auc['Mean_score'].values,
            'perm_f1_mean': df_perm_f1['Mean_score'].mean(),
            'perm_f1_all': df_perm_f1['Mean_score'].values,
            'perm_kappa_mean': df_perm_kappa['Mean_score'].mean(),
            'perm_kappa_all': df_perm_kappa['Mean_score'].values,
        }, ignore_index=True)



    mind_glmm = results_df.assign(
                        GLMM_p_corrected = lambda df: multipletests(df.Glmm_pval, method = 'fdr_bh')[1],
                        GLMM_significant = lambda df: np.select([(df.Glmm_pval < 0.05) & (df.GLMM_p_corrected < 0.05), (df.Glmm_pval < 0.05) & (df.GLMM_p_corrected > 0.05),
                                                    (df.Glmm_pval > 0.05) & (df.GLMM_p_corrected > 0.05)], ['p < 0.05 FDR corrected','p < 0.05 uncorrected', 'p > 0.05']),
        
                        AUC_p_corrected = lambda df: multipletests(df.AUC_P_val, method = 'fdr_bh')[1],
                        AUC_significant = lambda df: np.select([(df.AUC_P_val < 0.05) & (df.AUC_p_corrected < 0.05), (df.AUC_P_val < 0.05) & (df.AUC_p_corrected > 0.05),  
                                                    (df.AUC_P_val > 0.05) & (df.AUC_p_corrected > 0.05)], ['p < 0.05 FDR corrected','p < 0.05 uncorrected', 'p > 0.05']),
                        F1_p_corrected = lambda df: multipletests(df.F1_P_val, method = 'fdr_bh')[1],
                        F1_significant = lambda df: np.select([(df.F1_P_val < 0.05) & (df.F1_p_corrected < 0.05), (df.F1_P_val < 0.05) & (df.F1_p_corrected > 0.05),  
                                                    (df.F1_P_val > 0.05) & (df.F1_p_corrected > 0.05)], ['p < 0.05 FDR corrected','p < 0.05 uncorrected', 'p > 0.05']),
                        Kappa_p_corrected = lambda df: multipletests(df.Kappa_P_val, method = 'fdr_bh')[1],
                        Kappa_significant = lambda df: np.select([(df.Kappa_P_val < 0.05) & (df.Kappa_p_corrected < 0.05), (df.Kappa_P_val < 0.05) & (df.Kappa_p_corrected > 0.05),
                                                    (df.Kappa_P_val > 0.05) & (df.Kappa_p_corrected > 0.05)], ['p < 0.05 FDR corrected','p < 0.05 uncorrected', 'p > 0.05'])
                    )

    mind_glmm.to_csv(os.path.join(results_path,f'univariate_glmm_mind_perm_k{n_splits}.csv'))


    # %%
    # segment_mind_roc = segment_mind_roc.sort_values(by = 'AUC', ascending = False).head(10).append(segment_mind_roc.sort_values(by = 'AUC', ascending = False).tail(10))
    metrics = ['AUC', 'F1', 'Kappa']
    for metric in metrics:
        fig = px.scatter(mind_glmm.sort_values(by = f'{metric}_mean'),x = f'{metric}_mean', y = 'Marker', template = "plotly_white", symbol = f'{metric}_significant', 
                        symbol_sequence = ['circle-open','circle','hexagram' ],
        #                  color = 'significant',
                        color_discrete_sequence = [pink, green,orange, pink], 
                        
                        category_orders = {'{metric}_significant': ['p > 0.05','p < 0.05 uncorrected', 'p < 0.05 FDR corrected']},
                        labels = {f'{metric}_mean': f'{metric}', 'significant': 'Statistical Significance', 'markers':''}
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
                    # range = [0.45,0.80], 
                    tickfont = {"size": 20},
                ),
            yaxis = dict(
                tickfont = {"size": 20},
                autorange = False,    
                automargin = True,
                range = [-1,len(mind_glmm)],
                dtick = 1
                ),
            showlegend=True, 

        )

        # fig.show()

        fig.write_image(os.path.join(fig_path,f'univariate_glmm_mind_perm_{metric}_k{n_splits}.png'))
        fig.write_image(os.path.join(fig_path,f'univariate_glmm_mind_perm_{metric}_k{n_splits}.pdf'))


# %%


    results_df = pd.DataFrame(columns=['Marker', 'Log-Likelihood', 'AIC','Estimate', 'P_val', 
                                    'score_mean', 'score_std', 'score_sem', 'score_range', 'perm_score','perms_score_std','perms_score_all'])


    for marker in tqdm(df_mw.drop(['mind', 'mind_numeric', 'participant', 'segment'], axis = 1).columns, desc="Markers"):
        formula = f"mind_numeric ~ {marker} + (1|participant)"
        
        other_markers = np.setdiff1d(df_mw.drop(['mind', 'mind_numeric', 'participant', 'segment'], axis = 1).columns, marker)
        df_marker = df_mw.drop(other_markers, axis = 1).reset_index()
        
        # Fitting the LMER model
        base_model = Lmer(formula, data=df_mw, family="binomial")
        base_model.fit(verbose = False, summary = False)
        
        # Stratified KFold for ROC AUC
        # skf = StratifiedKFold(n_splits=5)
        X = df_marker[marker].values.reshape(-1, 1)
        y = df_marker['mind_numeric']
        scores = []
        perm_scores_all = []

        auc_scores = []
        f1_scores = []
        kappa_scores = []
        # Initialize the lists outside the loop
        perm_auc, perm_f1, perm_kappa = [], [], []
        
        # n_splits = 5
        group_kfold = GroupKFold(n_splits=n_splits)

        # Assuming df_mw['participant'] contains the participant IDs
        groups = df_marker['participant'].values

        for train_index, test_index in group_kfold.split(X, y, groups):
            df_train = df_marker.iloc[train_index]
            # df_train[marker] = StandardScaler().fit_transform(df_train[marker].values.reshape(-1, 1))
            df_test = df_marker.iloc[test_index]
            # df_test[marker] = StandardScaler().fit_transform(df_test[marker].values.reshape(-1, 1))
            y_test = y[test_index]
            # Save original labels
            original_labels = df_train['mind_numeric'].copy()
            
            model = Lmer(formula, data=df_train, family="binomial")
            model.fit(verbose = False, summary = False)
            

            predicted_probabilities = model.predict(df_test, use_rfx=True, verify_predictions=False)
            predicted_labels = [1 if prob > 0.5 else 0 for prob in predicted_probabilities]

            auc = roc_auc_score(y_test, predicted_probabilities)
            auc_scores.append(auc)
            f1 = f1_score(y_test, predicted_labels)
            f1_scores.append(f1)
            kappa = cohen_kappa_score(y_test, predicted_labels)
            kappa_scores.append(kappa)
            
            # Permutation test
            perm_scores = Parallel(n_jobs=-1)(
                delayed(perform_permutation)(formula, df_train.copy(), df_test, y_test, original_labels, label='mind_numeric')
                for _ in range(n_permutations)
            )

            # Append the mean of the scores
            perm_auc.append([score[0] for score in perm_scores])
            perm_f1.append([score[1] for score in perm_scores])
            perm_kappa.append([score[2] for score in perm_scores])
            

        # Convert 2D list to DataFrame
        df_perm_auc = pd.DataFrame(perm_auc).T
        df_perm_f1 = pd.DataFrame(perm_f1).T
        df_perm_kappa = pd.DataFrame(perm_kappa).T
        

        # Average across folds for each permutation run
        df_perm_auc['Mean_score'] = df_perm_auc.mean(axis=1)
        df_perm_f1['Mean_score'] = df_perm_f1.mean(axis=1)
        df_perm_kappa['Mean_score'] = df_perm_kappa.mean(axis=1)
            
        
        # Compute AUC statistics
        auc_mean = np.mean(auc_scores)
        auc_std = np.std(auc_scores)
        auc_pvalue = np.mean(df_perm_auc['Mean_score'] >= auc_mean)
        
        # Compute F1 statistics
        f1_mean = np.mean(f1_scores)
        f1_std = np.std(f1_scores)
        f1_pvalue = np.mean(df_perm_f1['Mean_score'] >= f1_mean)
        
        # Compute F1 statistics
        kappa_mean = np.mean(kappa_scores)
        kappa_std = np.std(kappa_scores)
        kappa_pvalue = np.mean(df_perm_kappa['Mean_score'] >= kappa_mean)
        
        
        # Save to DataFrame
        results_df = results_df.append({
            'Marker': marker,
            'Log-Likelihood': base_model.logLike,
            'AIC': base_model.AIC,
            'Glmm_pval': base_model.coefs['P-val'][0],
            'Estimate': base_model.coefs['Estimate'][0],
            'AUC_P_val': auc_pvalue,
            'AUC_mean': auc_mean,
            'AUC_std': auc_std,
            'F1_P_val': f1_pvalue,
            'F1_mean': f1_mean,
            'F1_std': f1_std,
            'Kappa_P_val': kappa_pvalue,
            'Kappa_mean': kappa_mean,
            'Kappa_std': kappa_std,
            'perm_auc_mean': df_perm_auc['Mean_score'].mean(),
            'perm_auc_all': df_perm_auc['Mean_score'].values,
            'perm_f1_mean': df_perm_f1['Mean_score'].mean(),
            'perm_f1_all': df_perm_f1['Mean_score'].values,
            'perm_kappa_mean': df_perm_kappa['Mean_score'].mean(),
            'perm_kappa_all': df_perm_kappa['Mean_score'].values,
        }, ignore_index=True)



    mw_glmm = results_df.assign(
                        GLMM_p_corrected = lambda df: multipletests(df.Glmm_pval, method = 'fdr_bh')[1],
                        GLMM_significant = lambda df: np.select([(df.Glmm_pval < 0.05) & (df.GLMM_p_corrected < 0.05), (df.Glmm_pval < 0.05) & (df.GLMM_p_corrected > 0.05),
                                                    (df.Glmm_pval > 0.05) & (df.GLMM_p_corrected > 0.05)], ['p < 0.05 FDR corrected','p < 0.05 uncorrected', 'p > 0.05']),
                        
                        AUC_p_corrected = lambda df: multipletests(df.AUC_P_val, method = 'fdr_bh')[1],
                        AUC_significant = lambda df: np.select([(df.AUC_P_val < 0.05) & (df.AUC_p_corrected < 0.05), (df.AUC_P_val < 0.05) & (df.AUC_p_corrected > 0.05),  
                                                    (df.AUC_P_val > 0.05) & (df.AUC_p_corrected > 0.05)], ['p < 0.05 FDR corrected','p < 0.05 uncorrected', 'p > 0.05']),
                        F1_p_corrected = lambda df: multipletests(df.F1_P_val, method = 'fdr_bh')[1],
                        F1_significant = lambda df: np.select([(df.F1_P_val < 0.05) & (df.F1_p_corrected < 0.05), (df.F1_P_val < 0.05) & (df.F1_p_corrected > 0.05),  
                                                    (df.F1_P_val > 0.05) & (df.F1_p_corrected > 0.05)], ['p < 0.05 FDR corrected','p < 0.05 uncorrected', 'p > 0.05']),
                        Kappa_p_corrected = lambda df: multipletests(df.Kappa_P_val, method = 'fdr_bh')[1],
                        Kappa_significant = lambda df: np.select([(df.Kappa_P_val < 0.05) & (df.Kappa_p_corrected < 0.05), (df.Kappa_P_val < 0.05) & (df.Kappa_p_corrected > 0.05),
                                                    (df.Kappa_P_val > 0.05) & (df.Kappa_p_corrected > 0.05)], ['p < 0.05 FDR corrected','p < 0.05 uncorrected', 'p > 0.05'])
                        )

    mw_glmm.to_csv(os.path.join(results_path,f'univariate_glmm_mw_perm_k{n_splits}.csv'))

    # %%
    # segment_mw_roc = segment_mw_roc.sort_values(by = 'score', ascending = False).head(10).append(segment_mw_roc.sort_values(by = 'score', ascending = False).tail(10))

    metrics = ['AUC', 'F1', 'Kappa']
    for metric in metrics:
        fig = px.scatter(mw_glmm.sort_values(by = f'{metric}_mean'),x = f'{metric}_mean', y = 'Marker', template = "plotly_white", symbol = f'{metric}_significant', 
                        symbol_sequence = ['circle-open','circle','hexagram' ],
        #                  color = 'significant',
                        color_discrete_sequence = [lblue, green,orange, pink], 
                        
                        category_orders = {'{metric}_significant': ['p > 0.05','p < 0.05 uncorrected', 'p < 0.05 FDR corrected']},
                        labels = {f'{metric}_score_mean': f'{metric}', 'significant': 'Statistical Significance', 'markers':''}
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
                    # range = [0.45,0.75], 
                    tickfont = {"size": 20},
                ),
            yaxis = dict(
                tickfont = {"size": 20},
                autorange = False,    
                automargin = True,
                range = [-1,len(mw_glmm)],
                dtick = 1
                ),
            showlegend=True, 

        )

        # fig.show()

        fig.write_image(os.path.join(fig_path,f'univariate_glmm_mw_perm_{metric}_k{n_splits}.png'))
        fig.write_image(os.path.join(fig_path,f'univariate_glmm_mw_perm_{metric}_k{n_splits}.pdf'))
        # fig.write_html(os.path.join(fig_path,f'univariate_glmm_mw_perm_{metric}.html'))


