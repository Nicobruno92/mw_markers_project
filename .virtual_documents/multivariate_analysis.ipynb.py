import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
    cross_val_score,
    GroupShuffleSplit,
    permutation_test_score,
    StratifiedKFold
)
from sklearn.ensemble import ExtraTreesClassifier

from utils import multivariate_classifier


epoch_type = "evoked"
# epoch_type = 'pseudo-rs'

all_participants = [
    "VP07",
    "VP08",
    "VP09",
    "VP10",
    "VP11",
    "VP12",
    "VP13",
    "VP14",
    "VP18",
    "VP19",
    "VP20",
    "VP22",
    "VP23",
    "VP24",
    "VP25",
    "VP26",
    "VP27",
    "VP28",
    "VP29",
    "VP30",
    "VP31",
    "VP32",
    "VP33",
    "VP35",
    "VP36",
    "VP37",
]

# path = "/media/nicolas.bruno/63f8a366-34b7-4896-a7ce-b5fb4ee78535/Nico/MW_eeg_data/minmarker/"  # icm-linux
path = '/Users/nicobruno/ownCloud/MW_eeg_data/minmarker/' #nico-mac


df = pd.DataFrame()

for i, v in enumerate(all_participants):
    participant = v

    folder = path + participant + "/"

    df_ = pd.read_csv(
        folder + participant + "_" + epoch_type + "_all_marker.csv", index_col=0
    )
    df_["participant"] = i
    df = df.append(df_)

df.to_csv("all_markers.csv")


df_markers = (
    df
#     .query("stimuli == 'go'")
#     .query("correct == 'correct'")
    .query("preproc == 'subtracted'")
    .query("prev_trial <= 4")
    .drop(
        [
            "stimuli",
            "correct",
            "prev_trial",
            "label",
            "events",
            "preproc",
            "epoch_type",
        ],
        axis=1,
    )
    .query("mind in ['on-task','dMW', 'sMW']")
    .groupby(["segment", "participant"])
    .filter(lambda x: len(x) > 1)
)

markers = [
    "wSMI",
    "p_e",
    "k",
    "b",
    "b_n",
    "g",
    "g_n",
    "t",
    "t_n",
    "d",
    "d_n",
    "a_n",
    "a",
    "CNV",
    "P1",
    "P3a",
    "P3b",
]


agg_dict = {k: ["mean", "std"] for k in markers}
agg_dict.update({k: "first" for k in df_markers.drop(markers, axis=1).columns})

df_mind = (
    df_markers.query("probe == 'PC'")
    .groupby(["segment", "participant"], as_index=False)
    .agg(agg_dict)
    #     .query("mind get_ipython().getoutput("= 'sMW'") #if you want to test against just one of the mw")
    .assign(mind2=lambda df: np.where(df.mind == "on-task", "on-task", "mw"))
)

df_mind.columns = df_mind.columns.map("_".join)

df_mind = df_mind.rename(
    columns={
        "participant_first": "participant",
        "probe_first": "probe",
        "mind_first": "mind",
        "segment_first": "segment",
        "mind2_": "mind2",
    }
).drop(["participant", "probe", "mind", "segment"], axis=1)


agg_dict = {k: "mean" for k in markers}
agg_dict.update({k: "first" for k in df_markers.drop(markers, axis=1).columns})

df_mind = (
    df_markers.query("probe == 'PC'")
    .groupby(["segment", "participant"], as_index=False)
    .agg(agg_dict)
    #     .query("mind get_ipython().getoutput("= 'sMW'") #if you want to test against just one of the mw")
    .assign(mind2=lambda df: np.where(df.mind == "on-task", "on-task", "mw"))
    .drop(
        ["participant", "probe", "mind", "segment"], axis=1
    )  # drop mind or mind2 also
)


df_auc = multivariate_classifier(data = df_mind, label = 'mind2', features = df_mind.drop('mind2', axis = 1).columns, model = 'SVM', permutation = True, n_permutations = 1000)


y, label = pd.factorize(df_mind['mind2'])
X = df_mind.drop('mind2', axis=1).astype('float32').values
n_estimators = 2000
doc_forest = make_pipeline(
    RobustScaler(),
    ExtraTreesClassifier(
        n_estimators=n_estimators, max_features=1, criterion='entropy',
        max_depth=4, random_state=42, class_weight='balanced'))

cv = GroupShuffleSplit(n_splits=50, train_size=0.8, test_size=0.2,
                       random_state=42)

aucs = cross_val_score(
    X=X, y=y, estimator=doc_forest,
    scoring='roc_auc', cv=cv, groups=np.arange(len(X)))

auc = pd.DataFrame(aucs, columns = ['auc'])
sns.catplot(x = 'auc', orient = 'h', data = auc, kind = 'violin')
plt.title(f'Mean = {np.mean(aucs)}; SD = {np.std(aucs)}')
plt.axvline(x = 0.5, linestyle = 'dashed')
plt.show()

doc_forest.fit(X, y)
variable_importance = doc_forest.steps[-1][-1].feature_importances_
sorter = variable_importance.argsort()

# shorten the names a bit.
var_names = list(df_mind.drop('mind2', axis = 1).columns)
var_names = [var_names[ii]for ii in sorter]

# let's plot it
plt.figure(figsize=(8, 6))
plt.scatter(
    doc_forest.steps[-1][-1].feature_importances_[sorter],
    np.arange(len(df_mind.drop('mind2', axis = 1).columns)))
plt.yticks(np.arange(len(df_mind.drop('mind2', axis = 1).columns)), var_names)
plt.subplots_adjust(left=.46)
plt.title('AUC={:0.3f}'.format(np.mean(aucs)))
plt.show()


agg_dict = {k:['mean','std'] for k in markers }
agg_dict.update({k:'first' for k in df_markers.drop(markers, axis=1).columns})

df_mw = (
    df_markers
    .query("probe == 'SC'")
    .query("mind get_ipython().getoutput("= 'on-task'")")
    .groupby(['segment', 'participant'], as_index = False).agg(agg_dict)
)

df_mw.columns = df_mw.columns.map("_".join)

df_mw  = (df_mw
            .rename(columns = {'participant_first':'participant', 'probe_first':'probe', 'mind_first':'mind', 'segment_first':'segment'})
            .drop(['participant', 'probe', 'segment'], axis = 1) 
           )


agg_dict = {k:'mean' for k in markers }
agg_dict.update({k:'first' for k in df_markers.drop(markers, axis=1).columns})

df_mw = (
    df_markers
    .query("probe == 'SC'")
    .query("mind get_ipython().getoutput("= 'on-task'")")
    .groupby(['segment', 'participant'], as_index = False).agg(agg_dict)
    .drop(['participant', 'probe', 'segment'], axis = 1)
)


df_auc = multivariate_classifier(data = df_mw, label = 'mind', features =  df_mw.drop('mind', axis = 1).columns, model = 'SVM', grid_search = False, plot = True, permutation = True, n_permutations = 1000)


y, label = pd.factorize(df_mw['mind'])
X = df_mw.drop('mind', axis=1).astype('float32').values
n_estimators = 2000
doc_forest = make_pipeline(
    RobustScaler(),
    ExtraTreesClassifier(
        n_estimators=n_estimators, max_features=1, criterion='entropy',
        max_depth=4, random_state=42, class_weight='balanced'))

cv = GroupShuffleSplit(n_splits=50, train_size=0.8, test_size=0.2,
                       random_state=42)

aucs = cross_val_score(
    X=X, y=y, estimator=doc_forest,
    scoring='roc_auc', cv=cv, groups=np.arange(len(X)))

auc = pd.DataFrame(aucs, columns = ['auc'])
sns.catplot(x = 'auc', orient = 'h', data = auc, kind = 'violin')
plt.title(f'Mean = {np.mean(aucs)}; SD = {np.std(aucs)}')
plt.axvline(x = 0.5, linestyle = 'dashed')
plt.show()



doc_forest.fit(X, y)
variable_importance = doc_forest.steps[-1][-1].feature_importances_
sorter = variable_importance.argsort()

# shorten the names a bit.
var_names = list(df_mw.drop('mind', axis = 1).columns)
var_names = [var_names[ii]for ii in sorter]

# let's plot it
plt.figure(figsize=(8, 6))
plt.scatter(
    doc_forest.steps[-1][-1].feature_importances_[sorter],
    np.arange(len(df_mw.drop('mind', axis = 1).columns)))
plt.yticks(np.arange(len(df_mw.drop('mind', axis = 1).columns)), var_names)
plt.subplots_adjust(left=.46)
plt.title('AUC={:0.3f}'.format(np.mean(aucs)))
plt.show()


df_mind = (
    df_markers
    .query("probe == 'PC'")
    .assign(
    mind2 = lambda df: np.where(df.mind == 'on-task', 'on-task', 'mw'))
#     .query("mind get_ipython().getoutput("= 'dMW'") #if you want to test against just one of the mw")
    .drop(['participant', 'probe', 'mind', 'segment'], axis = 1) # drop mind or mind2 also
)


df_auc = multivariate_classifier(data = df_mind, label = 'mind2', features = df_mind.drop('mind2', axis = 1).columns, model = 'SVM', permutation = True, n_permutations = 1000)


y, label = pd.factorize(df_mind['mind2'])
X = df_mind.drop('mind2', axis=1).astype('float32').values
n_estimators = 2000
doc_forest = make_pipeline(
    RobustScaler(),
    ExtraTreesClassifier(
        n_estimators=n_estimators, max_features=1, criterion='entropy',
        max_depth=4, random_state=42, class_weight='balanced'))

cv = GroupShuffleSplit(n_splits=50, train_size=0.8, test_size=0.2,
                       random_state=42)

aucs = cross_val_score(
    X=X, y=y, estimator=doc_forest,
    scoring='roc_auc', cv=cv, groups=np.arange(len(X)))

auc = pd.DataFrame(aucs, columns = ['auc'])
sns.catplot(x = 'auc', orient = 'h', data = auc, kind = 'violin')
plt.title(f'Mean = {np.mean(aucs)}; SD = {np.std(aucs)}')
plt.axvline(x = 0.5, linestyle = 'dashed')
plt.show()

markers = df_mind.drop('mind2', axis = 1).columns

doc_forest.fit(X, y)
variable_importance = doc_forest.steps[-1][-1].feature_importances_
sorter = variable_importance.argsort()

# shorten the names a bit.
var_names = list(markers)
var_names = [var_names[ii]for ii in sorter]

# let's plot it
plt.figure(figsize=(8, 6))
plt.scatter(
    doc_forest.steps[-1][-1].feature_importances_[sorter],
    np.arange(17))
plt.yticks(np.arange(17), var_names)
plt.subplots_adjust(left=.46)
plt.title('AUC={:0.3f}'.format(np.mean(aucs)))
plt.show()


df_mw = (
    df_markers
    .query("probe == 'SC'")
    .query("mind get_ipython().getoutput("= 'on-task'")")
    .drop(['participant', 'probe','segment'], axis = 1)
)


df_auc = multivariate_classifier(data = df_mw, label = 'mind', features =  df_mw.drop('mind', axis = 1).columns, model = 'SVM', grid_search = False, plot = True, permutation = True, n_permutations = 1000)


y, label = pd.factorize(df_mw['mind'])
X = df_mw.drop('mind', axis=1).astype('float32').values
n_estimators = 2000
doc_forest = make_pipeline(
    RobustScaler(),
    ExtraTreesClassifier(
        n_estimators=n_estimators, max_features=1, criterion='entropy',
        max_depth=4, random_state=42, class_weight='balanced'))

cv = GroupShuffleSplit(n_splits=50, train_size=0.8, test_size=0.2,
                       random_state=42)

aucs = cross_val_score(
    X=X, y=y, estimator=doc_forest,
    scoring='roc_auc', cv=cv, groups=np.arange(len(X)))

markers = df_mind.drop('mind2', axis = 1).columns

doc_forest.fit(X, y)
variable_importance = doc_forest.steps[-1][-1].feature_importances_
sorter = variable_importance.argsort()

# shorten the names a bit.
var_names = list(markers)
var_names = [var_names[ii]for ii in sorter]

# let's plot it
plt.figure(figsize=(8, 6))
plt.scatter(
    doc_forest.steps[-1][-1].feature_importances_[sorter],
    np.arange(17))
plt.yticks(np.arange(17), var_names)
plt.subplots_adjust(left=.46)
plt.title('AUC={:0.3f}'.format(np.mean(aucs)))
plt.show()

auc = pd.DataFrame(aucs, columns = ['auc'])
sns.catplot(x = 'auc', orient = 'h', data = auc, kind = 'violin')
plt.title(f'Mean = {np.mean(aucs)}; SD = {np.std(aucs)}')
plt.axvline(x = 0.5, linestyle = 'dashed')
plt.show()
