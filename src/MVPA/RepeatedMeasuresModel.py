import os
import numpy as np
import pandas as pd
from scipy.special import expit 

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier,ExtraTreesRegressor
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBRegressor

import optuna

from merf import MERF
from merf.utils import MERFDataGenerator
from imblearn.over_sampling import SMOTE

import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.io as pio
from plotly.subplots import make_subplots

class Optimization:
    def __init__(self, X, Z, y, groups, folds, results_path, database_name, study_name, merf = True, n_trials=100, data_augmentation = False, save_to_df=True):
        """
        Initialize the Optimization class.

        Args:
            X (pd.DataFrame): Feature matrix.
            Z (np.array): Random effects matrix.
            y (pd.Series): Target variable.
            groups (pd.Series): Grouping variable for random effects.
            folds (int): Number of folds for cross-validation.
            results_path (str): Path to save results.
            database_name (str): Database name for Optuna.
            study_name (str): Study name for Optuna.
            n_trials (int, optional): Number of trials for optimization. Defaults to 100.
            data_augmentation (bool, optional): Whether to use data augmentation for balancing. Defaults to False.
            save_to_df (bool, optional): Whether to save results to DataFrame. Defaults to True.
        """
        self.X = X
        self.Z = Z
        self.y = y
        self.groups = groups
        self.folds = folds
        self.results_path = results_path
        self.database_name = database_name
        self.study_name = study_name
        self.merf = merf
        self.n_trials = n_trials
        self.data_augmentation = data_augmentation
        self.save_to_df = save_to_df
        self.group_kfold = GroupKFold(n_splits=self.folds)
        self.trial_data = {}  # Global dictionary for storing trial data

        study_db_path = os.path.join(self.results_path, self.database_name)
        self.study = optuna.create_study(direction="maximize", study_name=self.study_name, storage=f'sqlite:///{self.database_name}', load_if_exists=True)
        self.study.optimize(self.objective, n_trials=self.n_trials,  n_jobs=1,)
        
        self.best_value = self.study.best_trial.value

        # Save the best trial's parameters to a file
        best_params_path = os.path.join(self.results_path, f'{self.study_name}_best_params.txt')
        with open(best_params_path, 'w') as file:
            for key, value in self.study.best_trial.params.items():
                file.write(f'{key}: {value}\n')

        # Save study trials to DataFrame if required
        if self.save_to_df:
            study_df = self.study.trials_dataframe()
            for trial_num in self.trial_data:
                study_df.loc[study_df.number == trial_num, 'fold_aucs'] = str(self.trial_data[trial_num]['fold_aucs'])
                study_df.loc[study_df.number == trial_num, 'feature_importances'] = str(self.trial_data[trial_num]['feature_importances'])
            study_df.sort_values('value', ascending=False).to_csv(os.path.join(self.results_path, f'{self.study_name}_opt_trials.csv'))

    def objective(self, trial):
        """
        Objective function for Optuna study.

        Args:
            trial (optuna.trial.Trial): Optuna trial object.

        Returns:
            float: Average AUC over all folds.
        """
        # Model configuration and cross-validation
        use_standardization = trial.suggest_categorical('use_standardization', [True, False])
        use_pca = trial.suggest_categorical('use_pca', [False])
        n_components = trial.suggest_int('n_components', 2, min(self.X.shape[1], self.X.shape[0])) if use_pca else None

        if self.merf:
            # MERF-specific hyperparameters
            # Model-specific hyperparameter space
            model = self.get_model(trial, model_type)
            model_type = trial.suggest_categorical('model_type', ['RandomForest',]) #'ExtraTrees', 'XGBoost'
            gll_early_stop_threshold = trial.suggest_float('gll_early_stop_threshold', 0.001, 0.1, log=True)
            max_iterations = trial.suggest_int('max_iterations', 10, 100)
            
        else:
            model = self.get_model(trial, model_type)


        scores = []
        feature_importances = np.zeros(self.X.shape[1])

        for train_index, test_index in self.group_kfold.split(self.X, self.y, self.groups.values):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            clusters_train, clusters_test = self.groups.iloc[train_index], self.groups.iloc[test_index]
            
            if self.data_augmentation:
            # Data augmentation for balancing
                X_train, y_train, Z_train, clusters_train = self.augment_data_with_smote(X_train, y_train, self.Z[train_index], clusters_train, trial)

            # Apply standardization and PCA if required
            X_train, X_test = self.preprocess_data(X_train, X_test, use_standardization, use_pca, n_components)

            if self.merf:
            # # Initialize and fit MERF model
                merf = MERF(fixed_effects_model=model, gll_early_stop_threshold=gll_early_stop_threshold, max_iterations=max_iterations)
                merf.fit(X_train, Z_train, clusters_train, y_train)
                y_pred_proba = expit(merf.predict(X_test, self.Z[test_index], clusters_test))  # Convert outputs to probabilities
                
            else:
                # Fit model
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_test)[:, 1]  # Assuming binary classification

            # Compute metrics
            optimal_threshold, auc = self.evaluate_model(y_test, y_pred_proba)
            scores.append(auc)
            
            # feature_importances += merf.fe_model.feature_importances_
            feature_importances += model.feature_importances_
            

        # Compute average feature importances over all folds
        feature_importances /= len(scores)  # Divide by the number of folds
        # Store trial data
        self.trial_data[trial.number] = {'fold_aucs': scores, 'feature_importances': feature_importances}

        # Return average AUC
        return np.mean(scores)

    def get_model(self, trial, model_type):
        """
        Returns a machine learning model based on the specified type and trial parameters.

        Args:
            trial (optuna.trial.Trial): Optuna trial object.
            model_type (str): Type of model (RandomForest, ExtraTrees, XGBoost).

        Returns:
            sklearn.base.BaseEstimator: Instantiated model with trial parameters.
        """
        if self.merf:   
            if model_type == 'RandomForest':
                return RandomForestRegressor(
                    n_estimators=trial.suggest_int('n_estimators', 10, 200),
                    max_depth=trial.suggest_int('max_depth', 2, 50),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
                    max_features=trial.suggest_float('max_features', 0.01, 1),
                    criterion=trial.suggest_categorical('criterion', ['squared_error', 'absolute_error', 'friedman_mse']),
                    bootstrap=trial.suggest_categorical('bootstrap', [True, False]),
                    min_impurity_decrease=trial.suggest_float('min_impurity_decrease', 0.0, 0.1),
                    random_state=42,
                    n_jobs=-1
                )
            elif model_type == 'ExtraTrees':
                return ExtraTreesRegressor(
                    n_estimators=trial.suggest_int('n_estimators', 10, 200),
                    max_depth=trial.suggest_int('max_depth', 2, 50),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
                    max_features=trial.suggest_float('max_features', 0.01, 1),
                    criterion=trial.suggest_categorical('criterion', ['squared_error', 'absolute_error', 'friedman_mse']),
                    bootstrap=trial.suggest_categorical('bootstrap', [True, False]),
                    min_impurity_decrease=trial.suggest_float('min_impurity_decrease', 0.0, 0.1),
                    random_state=42,
                    n_jobs=-1
                )
            elif model_type == 'XGBoost':
                return XGBRegressor(
                    n_estimators=trial.suggest_int('n_estimators', 50, 500),
                    max_depth=trial.suggest_int('max_depth', 3, 10),
                    learning_rate=trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                    gamma=trial.suggest_loguniform('gamma', 1e-8, 1.0),
                    min_child_weight=trial.suggest_int('min_child_weight', 1, 10),
                    subsample=trial.suggest_uniform('subsample', 0.5, 1.0),
                    colsample_bytree=trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
                    random_state=42,
                    n_jobs=-1
                )
        else:
            # Hyperparameters for RandomForest
            n_estimators = trial.suggest_int('n_estimators', 10, 200)
            max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
            max_features = trial.suggest_categorical('max_features', [ 'sqrt', 'log2'])
            bootstrap = trial.suggest_categorical('bootstrap', [True, False])
            criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
            
            return RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        bootstrap=bootstrap,
                        criterion=criterion,
                        class_weight='balanced',
                        random_state=42,
                        n_jobs=-1)


    def preprocess_data(self, X_train, X_test, use_standardization, use_pca, n_components):
        """
        Preprocesses the data based on the specified requirements (standardization, PCA).

        Args:
            X_train (pd.DataFrame): Training features.
            X_test (pd.DataFrame): Test features.
            use_standardization (bool): Whether to use standardization.
            use_pca (bool): Whether to use PCA.
            n_components (int): Number of components for PCA.

        Returns:
            tuple: Preprocessed training and test features.
        """
        if use_standardization:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        if use_pca:
            pca = PCA(n_components=n_components)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

        return X_train, X_test

    def augment_data_with_smote(self, X, y, Z, clusters, trial):
        # SMOTE configuration
        smote_k_neighbors = trial.suggest_int('smote_k_neighbors', 1, 4)

        augmented_X, augmented_y, augmented_Z, augmented_clusters = [], [], [], []
        single_sample_clusters = []

        for cluster_id in clusters.unique():
            cluster_mask = (clusters == cluster_id)
            X_cluster = X[cluster_mask]
            y_cluster = y[cluster_mask]
            Z_cluster = Z[cluster_mask]
            min_samples_in_cluster = min(X_cluster[y_cluster == label].shape[0] for label in y_cluster.unique())

            if min_samples_in_cluster > 1:
                k_neighbors = min(smote_k_neighbors, min_samples_in_cluster - 1)
                smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
                X_res, y_res = smote.fit_resample(X_cluster, y_cluster)
                augmented_X.append(X_res)
                augmented_y.append(y_res)

                augmented_clusters.extend([cluster_id] * len(X_res))
            else:
                smote = SMOTE(k_neighbors=0, random_state=42)
                X_res, y_res = smote.fit_resample(X_cluster, y_cluster)
                augmented_X.append(X_res)
                augmented_y.append(y_res)

                augmented_clusters.extend([cluster_id] * len(X_res))

        augmented_Z = np.ones((pd.concat(augmented_X).shape[0], 1))  # Random effects design matrix

        return pd.concat(augmented_X), pd.concat(augmented_y), augmented_Z, pd.Series(augmented_clusters)
    
    def evaluate_model(self, y_test, y_pred_proba):
        """
        Evaluates the model based on the test data and predicted probabilities.

        Args:
            y_test (pd.Series): Actual test labels.
            y_pred_proba (np.array): Predicted probabilities.

        Returns:
            tuple: Optimal threshold for classification and AUC score.
        """
        # Find optimal cutoff point based on F1 score
        thresholds = np.linspace(0, 1, 100)
        f1_scores = [f1_score(y_test, (y_pred_proba > t).astype(int)) for t in thresholds]
        optimal_threshold = thresholds[np.argmax(f1_scores)]

        # Compute AUC for probabilities
        auc = roc_auc_score(y_test, y_pred_proba)

        return optimal_threshold, auc
    
    def run_best_model(self, best_params_path):
        """
        Runs the model with the best hyperparameters found in the optimization process.

        Args:
            best_params_path (str): Path to the file containing the best parameters.
        """
        # Read the parameters from the file and store them in a dictionary
        best_params = {}
        with open(best_params_path, 'r') as file:
            for line in file:
                key, value = line.strip().split(': ')
                # Convert string representations of boolean to actual boolean values
                if value in ["True", "False"]:
                    value = value == "True"

                # Explicitly handle string and integer parameters
                if key in ['n_estimators', 'max_depth', 'min_samples_leaf', 
                        'max_iterations', 'min_samples_split', 'min_child_weight']:
                    best_params[key] = int(value)
                elif key in ['bootstrap', 'criterion', 'model_type', 
                            'use_standardization', 'use_pca']:
                    best_params[key] = value
                elif key == 'n_components' and value != 'None':
                    best_params[key] = int(value)
                else:
                    best_params[key] = float(value)


        scores = []
        optimal_cutoffs = []

        for train_index, test_index in self.group_kfold.split(self.X, self.y, self.groups.values):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            clusters_train, clusters_test = self.groups.iloc[train_index], self.groups.iloc[test_index]

            # Preprocess data
            X_train, X_test = self.preprocess_data(X_train, X_test, best_params.get('use_standardization', False), best_params.get('use_pca', False), best_params.get('n_components', None))

            # Create and configure the model with the best parameters
            model = self.get_model_from_params(best_params)
            merf = MERF(fixed_effects_model=model, gll_early_stop_threshold=best_params["gll_early_stop_threshold"], max_iterations=best_params["max_iterations"])
            
            merf.fit(X_train, self.Z[train_index], clusters_train, y_train)
            y_pred_proba = expit(merf.predict(X_test, self.Z[test_index], clusters_test))

            # Evaluate model
            optimal_threshold, auc = self.evaluate_model(y_test, y_pred_proba)
            scores.append(auc)
            optimal_cutoffs.append(optimal_threshold)

        # self.best_model_feature_importances = merf.fe_model.feature_importances_
        self.best_model_feature_importances = model.feature_importances_

        # Average AUC over all folds
        avg_auc = np.mean(scores)
        print(f"Scores per fold: {scores}")
        print(f"Average AUC: {avg_auc}")

    def get_model_from_params(self, best_params):
        """
        Returns a machine learning model based on the best parameters.

        Args:
            best_params (dict): Dictionary of the best parameters.

        Returns:
            sklearn.base.BaseEstimator: Configured model.
        """
        model_type = best_params.get('model_type', 'RandomForest')
        return self.get_model(optuna.trial.FixedTrial(best_params), model_type)
    

    def get_best_model_feature_importances(self, feature_names=None):
        """
        Retrieves the feature importances of the best model from the stored trial data.
        If not found, runs the best model to obtain them.

        Args:
            feature_names (list, optional): List of feature names. If None, numeric indices will be used.

        Returns:
            pd.Series or None: A Series containing feature importances of the best model.
        """

        best_trial = self.study.best_trial

        importances_df = pd.DataFrame()
        # Check if feature importances are available in the stored trial data
        if best_trial.number in self.trial_data:
            best_trial_data = self.trial_data[best_trial.number]
            if 'feature_importances' in best_trial_data:
                importances_df['features'] = feature_names if feature_names is not None else range(len(best_trial_data['feature_importances']))
                importances_df['importances'] =  pd.Series(best_trial_data['feature_importances'])
                return importances_df

        # If feature importances are not found, run the best model to get them
        print("Feature importances not found in trial data. Running best model to obtain feature importances.")
        self.run_best_model(os.path.join(self.results_path, self.study_name + '_best_params.txt'))
        if self.best_model_feature_importances is not None:
            importances_df['features'] = feature_names if feature_names is not None else range(len(self.best_model_feature_importances))
            importances_df['importances'] =  pd.Series(self.best_model_feature_importances)
            return importances_df

        print("Feature importances for the best model not found.")
        return None
    
    def plot_feat_importances(self, filename = 'test_fig', feature_names = None, color = px.colors.qualitative.Plotly[1], show = True, save_fig = True):
        """
        Plots the feature importances of the best model.
        
        Args:
            filename (str): The filename to save the plot image (default is 'test_fig').
            feature_names (list): List of feature names to include in the plot (default is None, which includes all features).
            color (str): The color of the markers in the plot (default is px.colors.qualitative.Plotly[1]).
            save_fig (bool): Whether to save the plot image (default is True).
        """
        
        importances_df = self.get_best_model_feature_importances(feature_names)
        
        fig = px.scatter(importances_df, x='importances', y='features', orientation='h',
                        title=f'Feat Imp MERF, AUC: {self.best_value:.3f}', template = "plotly_white",
                        color_discrete_sequence = [color],
                        labels = {'value':'Feature importance', 'features': 'Markers'}

                        )

        fig.update_traces(marker=dict(size = 8))

        fig.update_layout(
            width=650,
            height=900,
        #     autosize = True, 
            template = 'plotly_white',
                font=dict(
                family="Times new roman",
                size=20,
                color="black"
            ),
            xaxis = dict(
                    visible=True,
                    # range = [0.37,0.63], 
                    tickfont = {"size": 20},
                    title = 'Feature Importance'
                ),
            yaxis = dict(
                categoryorder =  'total ascending',
                tickfont = {"size": 20},
                # autorange = False,    
                automargin = True,
                range = [-1,len(importances_df)],
                dtick = 1
                ),
            showlegend=True, 

        )
        if show:
            fig.show()
        if save_fig:
            fig.write_image(filename)
            fig.write_image(filename)

