#!/usr/bin/env python3
"""
Script to define machine learning pipelines for classification tasks.
"""

import os
import logging

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
}

params_grids = {
    "Decision Tree": {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["auto", "sqrt", "log2"],
    },
    "Gradient Boosting": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 1],
        "max_depth": [3, 5, 10],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["auto", "sqrt", "log2"],
    },
    "K-Nearest Neighbors": {
        "n_neighbors": [3, 5, 10],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
}

def get_cv(n_splits=5, random_state=42):
    """Returns a StratifiedKFold cross-validation object."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

def process_data(X, use_preprocessing):
    """Apply standard scaling if specified."""
    if use_preprocessing:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        pipeline = Pipeline([("scaler", StandardScaler())])
        return pipeline.fit_transform(X)
    return X.values


def pipeline_baseline(X, y, use_preprocessing=False):
    """
    Baseline pipeline evaluates models with cross-validation and then fits each model.
    """
    cv = get_cv()
    results = {}
    saved_models = {}

    X_processed = process_data(X, use_preprocessing)
    for model_name, model in models.items():
        scores = cross_val_score(model, X_processed, y, cv=cv, scoring="accuracy")
        score_mean = scores.mean()
        logging.info(f"{model_name} Accuracy: {score_mean:.4f}")
        results[model_name] = score_mean
        model.fit(X_processed, y)
        saved_models[model_name] = model

    return results, saved_models

def pipeline_feature_selection(X, y, num_features, model_name, use_preprocessing=False):
    """
    Feature selection pipeline using SequentialFeatureSelector.
    """
    from sklearn.feature_selection import SequentialFeatureSelector

    base_model = models.get(model_name)
    if base_model is None:
        raise ValueError(f"Model '{model_name}' is not defined.")

    cv = get_cv()
    X_processed = process_data(X, use_preprocessing)
    selected_features_dict = {}

    for k in range(1, num_features + 1):
        logging.info(f"Processing {k} features")
        sfs = SequentialFeatureSelector(
            base_model,
            n_features_to_select=k,
            direction="forward",
            cv=cv,
            n_jobs=-1
        )
        sfs.fit(X_processed, y)
        selected_features = sfs.get_support(indices=True)
        X_selected = X_processed[:, selected_features]
        scores = cross_val_score(base_model, X_selected, y, cv=cv, scoring="accuracy", n_jobs=-1)
        score_mean = scores.mean()
        logging.info(f"Accuracy with {k} features: {score_mean:.4f}")
        result_dict = {
            "selected_features": selected_features,
            "accuracy": score_mean,
            "fitted_model": sfs.estimator
        }
        if hasattr(sfs.estimator, "feature_importances_"):
            result_dict["feature_importances"] = sfs.estimator.feature_importances_
        selected_features_dict[k] = result_dict

    return selected_features_dict

def pipeline_HP_search(X, y, model_name, use_preprocessing=False):
    """
    Hyperparameter search pipeline using GridSearchCV.
    """
    from sklearn.model_selection import GridSearchCV

    base_model = models.get(model_name)
    if base_model is None:
        raise ValueError(f"Model '{model_name}' is not defined.")

    cv = get_cv()
    X_processed = process_data(X, use_preprocessing)
    param_grid = params_grids.get(model_name)
    search = GridSearchCV(
        base_model,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )
    search.fit(X_processed, y)
    logging.info(f"{model_name} - Best parameters: {search.best_params_}")
    logging.info(f"{model_name} - Best accuracy: {search.best_score_:.4f}")

    return search.best_score_

def pipeline_feature_selection_HP_search(X, y, start_feature, num_features, model_name, use_preprocessing=False):
    """
    Combined feature selection and hyperparameter search pipeline.
    """
    from sklearn.feature_selection import SequentialFeatureSelector
    from sklearn.model_selection import GridSearchCV

    base_model = models.get(model_name)
    if base_model is None:
        raise ValueError(f"Model '{model_name}' is not defined.")

    cv = get_cv()
    X_processed = process_data(X, use_preprocessing)
    selected_features_dict = {}

    for k in range(1, num_features + 1):
        logging.info(f"Processing {k} features")
        sfs = SequentialFeatureSelector(
            base_model,
            n_features_to_select=k,
            direction="forward",
            cv=cv,
            n_jobs=-1
        )
        sfs.fit(X_processed, y)
        selected_features = sfs.get_support(indices=True)
        X_selected = X_processed[:, selected_features]
        model_instance = models.get(model_name)
        param_grid = params_grids.get(model_name)
        search = GridSearchCV(
            model_instance,
            param_grid=param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1
        )
        search.fit(X_selected, y)
        logging.info(f"{model_name} - Best parameters with {k} features: {search.best_params_}")
        logging.info(f"{model_name} - Best accuracy with {k} features: {search.best_score_:.4f}")
        result_dict = {
            "selected_features": selected_features,
            "accuracy": search.best_score_,
            "best_params": search.best_params_,
            "fitted_model": search.best_estimator_
        }
        if hasattr(search.best_estimator_, "feature_importances_"):
            result_dict["feature_importances"] = search.best_estimator_.feature_importances_
        selected_features_dict[k] = result_dict

    return selected_features_dict

def pipeline_unsupervised(X, n_clusters=2, use_preprocessing=False):
    """
    Unsupervised pipeline using KMeans and silhouette score.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    X_processed = process_data(X, use_preprocessing)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_processed)
    silhouette_avg = silhouette_score(X_processed, cluster_labels)
    logging.info(f"KMeans with {n_clusters} clusters - Silhouette score: {silhouette_avg:.4f}")

    return silhouette_avg, cluster_labels
