import pandas as pd
import numpy as np

from tqdm import tqdm 
import os

from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    GroupShuffleSplit,
    permutation_test_score,
    StratifiedKFold,
    KFold
)

import matplotlib.pyplot as plt
import seaborn as sns

import argparse

#############################
#### Classifier Function ####
#############################
def univariate_classifier(
    data, label, feature, model, permutation=False, n_permutations = 1000, perm_plot = False
):
    """
    data: dataframe with features and labels
    label: name of the column with the labels for the classification
    features: feaure or list of features corresponding to the columns of the data frame with the markers
    model: type of classifier model
    grid_search: if true it will apply grid search 5cv to find the best parameters of C and gamma, only for the SVM. Deafault: False
    """
    y, lbl = pd.factorize(data[label])
    X = data[feature].astype("float32").values.reshape(-1,1)
    
    if model == "SVM":
            steps = [
                ("scaler", StandardScaler()),
                ("SVM", SVC(C=0.001, gamma=0.1, kernel="rbf", probability=True)),
            ]
            pipe_cv = Pipeline(steps)

            cv = StratifiedKFold(10, shuffle=True, random_state = 42)

            aucs = cross_val_score(
                X=X,
                y=y,
                estimator=pipe_cv,
                scoring="roc_auc",
                cv=cv,
            )
            

    if model == 'forest':
            steps = [
            ("scaler", StandardScaler()),
                       ]
            n_estimators = 1000
            steps.append(('Forest',ExtraTreesClassifier(
                n_estimators=n_estimators, max_features='auto', criterion='entropy',
                max_depth=None, random_state=42, class_weight=None)))
            pipe_cv = Pipeline(steps)

            cv = StratifiedKFold(10, shuffle=True, random_state = 42)

            aucs = cross_val_score(
                X=X,
                y=y,
                estimator=pipe_cv,
                scoring="roc_auc",
                cv=cv,
            )

    tqdm.write(f'AUC {feature} = {np.mean(aucs)}')
            
    if permutation == True:
        score, perm_scores, pvalue = permutation_test_score(
            pipe_cv, X, y, scoring="roc_auc", cv=cv, n_permutations=n_permutations, n_jobs =-1
        )
            

        tqdm.write(f"p_value = {pvalue}")
        
        
        if perm_plot == True:

            plt.hist(perm_scores, bins=20, density=True)
            plt.axvline(score, ls="--", color="r")
            score_label = (
                f"Score on original\ndata: {score:.2f}\n" f"(p-value: {pvalue:.3f})"
            )
            plt.text(score, np.max(perm_scores), score_label, fontsize=12)
            plt.xlabel("Accuracy score")
            plt.ylabel("Probability")
            plt.show()
        
        return aucs, pvalue
    
    else:
        return aucs, 0


#######################
#### Main Function ####
#######################  
def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Univariate RF classification of EEG markers')
    parser.add_argument('data', type=str, metavar='markers.csv',
                        help='input data must be .csv Ex: all_markers.csv')
    parser.add_argument('--output_path', type=str, default='./',
                        metavar='out_path', help='output path')
    parser.add_argument('--contrast', type=str, default='./',
                        metavar='contrast', help='classification contrast ["mind" or "mw"]')
    args = parser.parse_args()
    
    contrast = args.contrast
    
    
    ######################
    #### Load markers ####
    ######################  
    df_markers = pd.read_csv(args.data)

    markers = ['wSMI_1', 'wSMI_2', 'wSMI_4', 'wSMI_8', 'p_e_1', 'p_e_2',
        'p_e_4', 'p_e_8', 'k', 'se','msf', 'sef90', 'sef95', 'b', 'b_n', 'g',
        'g_n', 't', 't_n', 'd', 'd_n', 'a_n', 'a', 'CNV', 'P1', 'P3a', 'P3b',]  
    agg_dict = {k:['mean', 'std'] for k in markers }
    agg_dict.update({k:'first' for k in df_markers.drop(markers, axis=1).columns})
    
    
    if contrast == 'mind':

        df_mind = (
            df_markers
            .query("probe == 'PC'")
            .groupby(['segment', 'participant'], as_index = False).agg(agg_dict)
            .assign(
            mind2 = lambda df: np.where(df.mind == 'on-task', 'on-task', 'mw'))
        )

        df_mind.columns = df_mind.columns.map("_".join)

        df_mind  = (df_mind
                    .rename(columns = {'participant_first':'participant', 'probe_first':'probe', 'mind_first':'mind', 'segment_first':'segment', 'mind2_':'mind2'})
        #             .query("mind != 'dMW'") #if you want to test against just one of the mw            
                    .drop(['participant', 'probe', 'mind', 'segment'], axis = 1) 
                )

        ########################
        #### Run classifier ####
        ########################
        AUC = pd.DataFrame()
        pvalues = {}
        for i in tqdm(df_mind.drop('mind2', axis = 1).columns):

            AUC[i], pvalues[i] =  univariate_classifier(
                data=df_mind, label='mind2', feature=i, 
                model='forest', permutation=True, n_permutations=1000)

            
        sns.catplot(data = AUC, kind = 'box', orient = 'h')
        plt.axvline(x = 0.5, linestyle = 'dashed')
        plt.savefig(os.path.join(args.output_path, 'catplot.png'))

        p_df = pd.DataFrame.from_dict(
            pvalues, orient='index',
            columns=['p_value']
        ).reset_index().rename(columns ={'index': 'markers'})
        
        classifier_mind = (
            AUC.reset_index()
                .melt(id_vars = ['index'], var_name = 'markers', value_name = 'AUC')
                .drop('index', axis = 1)
                .merge(p_df, on = 'markers', how = 'inner')
                .assign(p_corrected = lambda df: multipletests(df.p_value, method = 'fdr_bh')[1],
                        significant = lambda df: np.select([(df.p_value < 0.05) & (df.p_corrected < 0.05), (df.p_value < 0.05) & (df.p_corrected > 0.05),  
                                                (df.p_value > 0.05) & (df.p_corrected > 0.05)], ['p < 0.05 FDR corrected','p < 0.05 uncorrected', 'p > 0.05']))
        )

        # File with all the results
        classifier_mind.to_csv(os.path.join(args.output_path, 'output_mind.csv'))
    
    if contrast = 'mw':  
        agg_dict = {k:['mean', 'std'] for k in markers }
        agg_dict.update({k:'first' for k in df_markers.drop(markers, axis=1).columns})

        df_mw = (
            df_markers
            .query("probe == 'SC'")
            .query("mind != 'on-task'")
            .groupby(['segment', 'participant'], as_index = False).agg(agg_dict)
            )

        df_mw.columns = df_mw.columns.map("_".join)

        df_mw  = (df_mw
                    .rename(columns = {'participant_first':'participant', 'probe_first':'probe', 'mind_first':'mind', 'segment_first':'segment'})
                    .drop(['participant', 'probe', 'segment'], axis = 1) 
                ) 
        ########################
        #### Run classifier ####
        ########################
        AUC = pd.DataFrame()
        pvalues = {}
        for i in tqdm(df_mw.drop('mind', axis = 1).columns):
            AUC[i], pvalues[i] =  univariate_classifier(
            data= df_mw, label = 'mind', feature = i, model = 'SVM', grid_search=False, permutation=True, n_permutations = 1000)
            
        sns.catplot(data = AUC, kind = 'box', orient = 'h')
        plt.axvline(x = 0.5, linestyle = 'dashed')
        plt.savefig(os.path.join(args.output_path, 'catplot.png'))
            
        p_df =pd.DataFrame.from_dict(pvalues, orient = 'index', columns = ['p_value']).reset_index().rename(columns ={'index': 'markers'})
        classifier_mw = (AUC.reset_index().melt(id_vars = ['index'], var_name = 'markers', value_name = 'AUC')
                    .drop('index', axis = 1)
                    .merge(p_df, on = 'markers', how = 'inner')
                    .assign(p_corrected = lambda df: multipletests(df.p_value, method = 'fdr_bh')[1],
                            significant = lambda df: np.select([(df.p_value < 0.05) & (df.p_corrected < 0.05), (df.p_value < 0.05) & (df.p_corrected > 0.05),  
                                                        (df.p_value > 0.05) & (df.p_corrected > 0.05)], ['p < 0.05 FDR corrected','p < 0.05 uncorrected', 'p > 0.05']))
                )
        # File with all the results
        classifier_mw.to_csv(os.path.join(args.output_path, 'output_mw.csv'))
    
    
if __name__ == '__main__':
    main()

