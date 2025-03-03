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

def pipeline_baseline(X, y, scoring="accuracy"):
    """
    Baseline pipeline evaluates models with cross-validation and then fits each model.
    """
    cv = get_cv()
    results = {}
    saved_models = {}

    for model_name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        score_mean = scores.mean()
        logging.info(f"{model_name} {scoring}: {score_mean:.4f}")
        results[model_name] = score_mean
        model.fit(X, y)
        saved_models[model_name] = model

    return results, saved_models

def pipeline_feature_selection(X, y, num_features, model_name, scoring="accuracy"):
    """
    Feature selection pipeline using SequentialFeatureSelector.
    """
    from sklearn.feature_selection import SequentialFeatureSelector

    cv = get_cv()
    selected_features_dict = {}

    for k in range(1, num_features + 1):
        logging.info(f"Processing {k} features")
        base_model = models.get(model_name)
        if base_model is None:
            raise ValueError(f"Model '{model_name}' is not defined.")
        sfs = SequentialFeatureSelector(
            base_model,
            n_features_to_select=k,
            direction="forward",
            cv=cv,
            n_jobs=-1,
            scoring=scoring
        )
        sfs.fit(X, y)
        selected_features = sfs.get_support(indices=True)
        X_selected = X[:, selected_features]
        scores = cross_val_score(base_model, X_selected, y, cv=cv, scoring=scoring, n_jobs=-1)
        score_mean = scores.mean()
        logging.info(f"{scoring} with {k} features: {score_mean:.4f}")
        result_dict = {
            "selected_features": selected_features,
            f"{scoring}": score_mean,
            "fitted_model": sfs.estimator
        }
        if hasattr(sfs.estimator, "feature_importances_"):
            result_dict["feature_importances"] = sfs.estimator.feature_importances_
        selected_features_dict[k] = result_dict

    return selected_features_dict

def pipeline_HP_search(X, y, model_name, scoring="accuracy"):
    """
    Hyperparameter search pipeline using GridSearchCV.
    """
    from sklearn.model_selection import GridSearchCV

    base_model = models.get(model_name)
    if base_model is None:
        raise ValueError(f"Model '{model_name}' is not defined.")

    cv = get_cv()
    param_grid = params_grids.get(model_name)
    search = GridSearchCV(
        base_model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    search.fit(X, y)
    best_score = search.best_score_
    best_params = search.best_params_
    logging.info(f"{model_name} - Best parameters: {best_params}")
    logging.info(f"{model_name} - Best {scoring}: {best_score:.4f}")

    return best_score, best_params, search.best_estimator_

def pipeline_feature_selection_HP_search(X, y, num_features, model_name, scoring="accuracy"):
    """
    Combined feature selection and hyperparameter search pipeline.
    Leverages pipeline_feature_selection and pipeline_HP_search.
    """
    feature_selection_results = pipeline_feature_selection(X, y, num_features, model_name, scoring)
    combined_results = {}

    for k, fs_result in feature_selection_results.items():
        logging.info(f"Performing hyperparameter search on {k} selected features")
        selected_features = fs_result["selected_features"]
        X_selected = X[:, selected_features]
        best_score, best_params, best_estimator = pipeline_HP_search(X_selected, y, model_name, scoring)
        logging.info(f"{model_name} - Best parameters with {k} features after HP search: {best_params}")
        logging.info(f"{model_name} - Best {scoring} with {k} features after HP search: {best_score:.4f}")
        result_dict = {
            "selected_features": selected_features,
            f"{scoring}": best_score,
            "best_params": best_params,
            "fitted_model": best_estimator
        }
        if hasattr(best_estimator, "feature_importances_"):
            result_dict["feature_importances"] = best_estimator.feature_importances_
        combined_results[k] = result_dict

    return combined_results

def pipeline_unsupervised(X, n_clusters=2):
    """
    Unsupervised pipeline using KMeans and silhouette score.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    logging.info(f"KMeans with {n_clusters} clusters - Silhouette score: {silhouette_avg:.4f}")

    return silhouette_avg, cluster_labels
