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

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from pymer4.models import Lmer

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from joblib import Parallel, delayed

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
def perform_permutation(formula, df_train, df_test, y_test):
    y_perm = np.random.permutation(df_train['mind2_numeric'])
    model_perm = Lmer(formula, data=df_train.assign(mind2_numeric=y_perm), family="binomial")
    model_perm.fit(verbose=False, summary = False)

    # Make predictions and compute AUC for permuted labels
    predicted_probabilities_perm = model_perm.predict(df_test, use_rfx=True, verify_predictions=False)
    perm_auc = roc_auc_score(y_test, predicted_probabilities_perm)
    return perm_auc

# Execute the loop in parallel
n_permutations = 1  # You can adjust this

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


# %%
results_df = pd.DataFrame(columns=['Marker', 'Log-Likelihood', 'AIC', 'P_val', 
                                   'AUC_mean', 'AUC_std', 'AUC_sem', 'AUC_range'])


for marker in tqdm(df_mind.drop(['mind2', 'mind2_numeric', 'participant', 'segment'], axis = 1).columns, desc="Markers"):
    formula = f"mind2_numeric ~ {marker} + (1|participant)"
    
    # Fitting the LMER model
    model = Lmer(formula, data=df_mind, family="binomial")
    model.fit(verbose = False, summary = False)
    
    # Stratified KFold for ROC AUC
    skf = StratifiedKFold(n_splits=5)
    X = df_mind[marker].values.reshape(-1, 1)
    y = df_mind['mind2_numeric']
    auc_scores = []
    perm_auc_scores_all = []
    
    for train_index, test_index in skf.split(X, y):
        df_train = df_mind.iloc[train_index]
        df_test = df_mind.iloc[test_index]
        y_test = y[test_index]
        model = Lmer(formula, data=df_train, family="binomial")
        model.fit(verbose = False, summary = False)
        

        predicted_probabilities = model.predict(df_test, use_rfx=True, verify_predictions=False)
        
        auc = roc_auc_score(y_test, predicted_probabilities)
        auc_scores.append(auc)
        
        perm_auc_scores = Parallel(n_jobs=-1)(
                                delayed(perform_permutation)(formula, df_train, df_test, y_test)
                                for _ in range(n_permutations)
                            )
        
        # Add the list of scores for this fold to the master list
        perm_auc_scores_all.append(perm_auc_scores)

    # Convert 2D list to DataFrame
    df_perm = pd.DataFrame(perm_auc_scores_all).T

    # Average across folds for each permutation run
    df_perm['Mean_AUC'] = df_perm.mean(axis=1)
        
    
    
    # Compute AUC statistics
    auc_mean = np.mean(auc_scores)
    auc_std = np.std(auc_scores)
    auc_sem = auc_std / np.sqrt(len(auc_scores))
    auc_range = np.ptp(auc_scores)
    
    
    pvalue = np.mean(df_perm['Mean_AUC'] >= auc_mean)
    
    # Save to DataFrame
    results_df = results_df.append({
        'Marker': marker,
        'Log-Likelihood': model.logLike,
        'AIC': model.AIC,
        'P_val': pvalue,
        'AUC_mean': auc_mean,
        'AUC_std': auc_std,
        'AUC_sem': auc_sem,
        'AUC_range': auc_range
    }, ignore_index=True)




mind_glmm = results_df.assign(
                    p_corrected = lambda df: multipletests(df.P_val, method = 'fdr_bh')[1],
                    significant = lambda df: np.select([(df.P_val < 0.05) & (df.p_corrected < 0.05), (df.P_val < 0.05) & (df.p_corrected > 0.05),  
                                                 (df.P_val > 0.05) & (df.p_corrected > 0.05)], ['p < 0.05 FDR corrected','p < 0.05 uncorrected', 'p > 0.05'])
                   )

mind_glmm.to_csv(os.path.join(results_path,'univariate_glmm_mind_perm.csv'))


# %%
# segment_mind_roc = segment_mind_roc.sort_values(by = 'AUC', ascending = False).head(10).append(segment_mind_roc.sort_values(by = 'AUC', ascending = False).tail(10))

fig = px.scatter(mind_glmm.sort_values(by = 'AUC_mean'),x = 'AUC_mean', y = 'Marker', template = "plotly_white", symbol = 'significant', 
                 symbol_sequence = ['circle-open','circle','hexagram' ],
#                  color = 'significant',
                 color_discrete_sequence = [pink, green,orange, pink], 
                 
                 category_orders = {'significant': ['p > 0.05','p < 0.05 uncorrected', 'p < 0.05 FDR corrected']},
                 labels = {'AUC_mean': 'TUT>OT                      TUT<OT', 'significant': 'Statistical Significance', 'markers':''}
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
            range = [0.45,0.80], 
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

fig.show()

fig.write_image(os.path.join(fig_path,'univariate_glmm_mind_perm.png'))
fig.write_image(os.path.join(fig_path,'univariate_glmm_mind_perm.pdf'))

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

# %%
results_df = pd.DataFrame(columns=['Marker', 'Log-Likelihood', 'AIC', 'P_val', 
                                   'AUC_mean', 'AUC_std', 'AUC_sem', 'AUC_range'])


for marker in tqdm(df_mw.drop(['mind', 'mind_numeric', 'participant', 'segment'], axis = 1).columns, desc="Markers"):
    formula = f"mind_numeric ~ {marker} + (1|participant)"
    
    # Fitting the LMER model
    model = Lmer(formula, data=df_mw, family="binomial")
    model.fit(verbose = False, summary = False)
    
    # Stratified KFold for ROC AUC
    skf = StratifiedKFold(n_splits=5)
    X = df_mind[marker].values.reshape(-1, 1)
    y = df_mind['mind_numeric']
    auc_scores = []
    perm_auc_scores_all = []
    
    for train_index, test_index in skf.split(X, y):
        df_train = df_mw.iloc[train_index]
        df_test = df_mw.iloc[test_index]
        y_test = y[test_index]
        model = Lmer(formula, data=df_train, family="binomial")
        model.fit(verbose = False, summary = False)
        

        predicted_probabilities = model.predict(df_test, use_rfx=True, verify_predictions=False)
        
        auc = roc_auc_score(y_test, predicted_probabilities)
        auc_scores.append(auc)
        
        perm_auc_scores = Parallel(n_jobs=-1)(
                                delayed(perform_permutation)(formula, df_train, df_test, y_test)
                                for _ in range(n_permutations)
                            )
        
        # Add the list of scores for this fold to the master list
        perm_auc_scores_all.append(perm_auc_scores)

    # Convert 2D list to DataFrame
    df_perm = pd.DataFrame(perm_auc_scores_all).T

    # Average across folds for each permutation run
    df_perm['Mean_AUC'] = df_perm.mean(axis=1)
        
    
    
    # Compute AUC statistics
    auc_mean = np.mean(auc_scores)
    auc_std = np.std(auc_scores)
    auc_sem = auc_std / np.sqrt(len(auc_scores))
    auc_range = np.ptp(auc_scores)
    
    
    pvalue = np.mean(df_perm['Mean_AUC'] >= auc_mean)
    
    # Save to DataFrame
    results_df = results_df.append({
        'Marker': marker,
        'Log-Likelihood': model.logLike,
        'AIC': model.AIC,
        'P_val': pvalue,
        'AUC_mean': auc_mean,
        'AUC_std': auc_std,
        'AUC_sem': auc_sem,
        'AUC_range': auc_range
    }, ignore_index=True)



mw_glmm = results_df.assign(
                    p_corrected = lambda df: multipletests(df.P_val, method = 'fdr_bh')[1],
                    significant = lambda df: np.select([(df.P_val < 0.05) & (df.p_corrected < 0.05), (df.P_val < 0.05) & (df.p_corrected > 0.05),  
                                                (df.P_val > 0.05) & (df.p_corrected > 0.05)], ['p < 0.05 FDR corrected','p < 0.05 uncorrected', 'p > 0.05'])
                    )

mw_glmm.to_csv(os.path.join(results_path,'univariate_glmm_mw_perm.csv'))


# %%
mw_glmm = pd.read_csv(os.path.join(results_path, 'univariate_glmm_mw.csv'))

# segment_mw_roc = segment_mw_roc.sort_values(by = 'AUC', ascending = False).head(10).append(segment_mw_roc.sort_values(by = 'AUC', ascending = False).tail(10))

fig = px.scatter(mw_glmm.sort_values(by = 'AUC_mean'),x = 'AUC_mean', y = 'Marker', template = "plotly_white", symbol = 'significant', 
                 symbol_sequence = ['circle-open','circle','hexagram' ],
#                  color = 'significant',
                 color_discrete_sequence = [lblue, green,orange, pink], 
                 
                 category_orders = {'significant': ['p > 0.05','p < 0.05 uncorrected', 'p < 0.05 FDR corrected']},
                 labels = {'AUC': 'sTUT>dTUT              sTUT<dTUT', 'significant': 'Statistical Significance', 'markers':''}
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
        range = [-1,len(mw_glmm)],
        dtick = 1
        ),
    showlegend=True, 

)

fig.show()

fig.write_image(os.path.join(fig_path,'univariate_glmm_mw_perm.png'))
fig.write_image(os.path.join(fig_path,'univariate_glmm_mw_perm.pdf'))
fig.write_html(os.path.join(fig_path,'univariate_glmm_mw_perm.html'))


