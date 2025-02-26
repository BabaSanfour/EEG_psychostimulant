#!/usr/bin/env python3
"""
Script to run machine learning pipelines for classification tasks.
"""

import os
import pickle
import argparse
import logging
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config import data_dir, results_dir
from ml_pipelines.pipelines import (
    pipeline_baseline,
    pipeline_feature_selection,
    pipeline_HP_search,
    pipeline_feature_selection_HP_search,
    pipeline_unsupervised,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def load_and_preprocess_features(analysis_type):
    """Load features CSV, filter based on age group and preprocess columns."""
    features_file = os.path.join(data_dir, "csv", "features_with_age_group.csv")
    features = pd.read_csv(features_file)
    if analysis_type == "adolescent":
        logging.info("Running analysis for adolescent group.")
        features = features[features["age_group"] == "adolescent"]
    elif analysis_type == "child":
        logging.info("Running analysis for child group.")
        features = features[features["age_group"] == "child"]
    else:
        logging.info("Running analysis for all ages.")
    drop_cols = ["dataset", "id", "age", "sex", "task", "subject", "age_group"]
    features = features.drop(columns=drop_cols)
    features["group"] = features["group"].replace({"PAT": 1, "CTR": 0})
    return features

def get_clean_features(columns):
    """Returns a list of clean feature names based on column names."""
    seen = set()
    clean_features = []
    for col in columns:
        feat = col.split(".spaces-")[0].replace("feature-", "")
        if feat not in seen and feat != "group":
            seen.add(feat)
            clean_features.append(feat)
    return clean_features

def get_columns_for_feature(columns, feature):
    """Return list of columns that contain the feature name."""
    return [col for col in columns if feature in col]

def save_results(results, fname):
    """Save results to a pickle file."""
    with open(os.path.join(results_dir, fname), "wb") as f:
        pickle.dump(results, f)

def run_analysis1(features, analysis_type):
    """Single feature HP search per sensor."""
    X = features.drop("group", axis=1)
    y = features["group"]
    model_results = {}
    for model in ["Decision Tree", "Random Forest", "Gradient Boosting", "K-Nearest Neighbors"]:
        logging.info(f"Running pipeline for model: {model}")
        feature_results = {}
        for feature in X.columns:
            logging.info(f"Running pipeline for feature: {feature}")
            acc = pipeline_HP_search(X[[feature]], y, model)
            feature_results[feature] = acc
        model_results[model] = feature_results
    fname = f"single_feature_HP_results_{analysis_type}.pkl"
    save_results(model_results, fname)
    logging.info("Done analysis 1!")


def run_analysis2(features, analysis_type):
    """Single feature HP search using all sensors (grouped by feature)."""
    feature_results = {}
    columns = features.columns
    clean_features = get_clean_features(columns)
    for feature in clean_features:
        select_cols = get_columns_for_feature(columns, feature)
        X = features[select_cols]
        y = features["group"]
        logging.info(f"Running classification for feature: {feature}")
        models_result = {}
        for model in ["Decision Tree", "Random Forest", "Gradient Boosting", "K-Nearest Neighbors"]:
            logging.info(f"Running HP search for model: {model}")
            models_result[model] = pipeline_HP_search(X, y, model)
        feature_results[feature] = models_result
    fname = f"single_feature_all_sensors_HP_results_{analysis_type}.pkl"
    save_results(feature_results, fname)
    logging.info("Done analysis 2!")


def run_analysis3(features, analysis_type):
    """Sensor selection per feature using feature selection."""
    feature_results = {}
    columns = features.columns
    clean_features = get_clean_features(columns)
    for feature in clean_features:
        select_cols = get_columns_for_feature(columns, feature)
        X = features[select_cols]
        y = features["group"]
        logging.info(f"Running feature selection for feature: {feature}")
        models_result = {}
        for model in ["Decision Tree", "Random Forest", "Gradient Boosting", "K-Nearest Neighbors"]:
            logging.info(f"Running pipeline for model: {model}")
            models_result[model] = pipeline_feature_selection(X, y, 20, model)
        feature_results[feature] = models_result
    fname = f"sensor_selection_per_feature_{analysis_type}.pkl"
    save_results(feature_results, fname)
    logging.info("Done analysis 3!")


def run_analysis4(features, analysis_type):
    """Sensor selection per feature followed by HP search."""
    feature_results = {}
    columns = features.columns
    clean_features = get_clean_features(columns)
    for feature in clean_features:
        select_cols = get_columns_for_feature(columns, feature)
        X = features[select_cols]
        y = features["group"]
        logging.info(f"Running selection + HP search for feature: {feature}")
        models_result = {}
        for model in ["Decision Tree", "Random Forest", "Gradient Boosting", "K-Nearest Neighbors"]:
            logging.info(f"Running pipeline for model: {model}")
            models_result[model] = pipeline_feature_selection_HP_search(X, y, 0, 7, model)
        feature_results[feature] = models_result
    fname = f"sensor_selection_per_feature_HP_search_{analysis_type}.pkl"
    save_results(feature_results, fname)
    logging.info("Done analysis 4!")


def run_analysis5(features, analysis_type):
    """Baseline multi-features using all sensors."""
    X = features.drop("group", axis=1)
    y = features["group"]
    results, saved_models = pipeline_baseline(X, y)
    combined_results = {"results": results, "models": saved_models}
    fname = f"baseline_results_all_feat_all_sensors_{analysis_type}.pkl"
    save_results(combined_results, fname)
    logging.info("Done analysis 5!")


def run_analysis6(features, analysis_type):
    """Multi-features with HP search for all models."""
    X = features.drop("group", axis=1)
    y = features["group"]
    model_results = {}
    for model in ["Decision Tree", "Random Forest", "Gradient Boosting", "K-Nearest Neighbors"]:
        logging.info(f"Running HP search for model: {model}")
        model_results[model] = pipeline_HP_search(X, y, model)
    fname = f"HP_results_all_feat_all_sensors_{analysis_type}.pkl"
    save_results(model_results, fname)
    logging.info("Done analysis 6!")


def run_analysis7(features, analysis_type):
    """Multi-feature feature selection (up to 40 features)."""
    X = features.drop("group", axis=1)
    y = features["group"]
    models_result = {}
    for model in ["Decision Tree", "Random Forest", "Gradient Boosting", "K-Nearest Neighbors"]:
        logging.info(f"Running feature selection for model: {model}")
        models_result[model] = pipeline_feature_selection(X, y, 40, model)
    fname = f"feature_selection_all_feat_all_sensors_{analysis_type}.pkl"
    save_results(models_result, fname)
    logging.info("Done analysis 7!")


def run_analysis8(features, analysis_type):
    """Multi-feature feature selection (up to 40 features) with HP search."""
    X = features.drop("group", axis=1)
    y = features["group"]
    models_result = {}
    for model in ["Decision Tree", "Random Forest", "Gradient Boosting", "K-Nearest Neighbors"]:
        logging.info(f"Running feature selection + HP search for model: {model}")
        models_result[model] = pipeline_feature_selection_HP_search(X, y, 0, 40, model)
    fname = f"feature_selection_HP_search_all_feat_all_sensors_{analysis_type}.pkl"
    save_results(models_result, fname)
    logging.info("Done analysis 8!")


def run_analysis9(features, analysis_type):
    """Unsupervised learning using all features and sensors."""
    X = features.drop("group", axis=1)
    clusters_results = {}
    for n_clusters in range(2, 11):
        logging.info(f"Running unsupervised pipeline for {n_clusters} clusters")
        silhouette, cluster_labels = pipeline_unsupervised(X, n_clusters)
        clusters_results[n_clusters] = {
            "silhouette": silhouette,
            "cluster_labels": cluster_labels,
        }
    fname = f"unsupervised_single_feature_all_sensors_{analysis_type}.pkl"
    save_results(clusters_results, fname)
    logging.info("Done analysis 9!")


def run_analysis10(features, analysis_type):
    """Unsupervised learning for each clean feature using all sensors."""
    feature_results = {}
    columns = features.columns
    clean_features = get_clean_features(columns)
    for feature in clean_features:
        select_cols = get_columns_for_feature(columns, feature)
        X = features[select_cols]
        cluster_results = {}
        for n_clusters in range(2, 11):
            logging.info(f"Running unsupervised for feature '{feature}' with {n_clusters} clusters")
            silhouette, cluster_labels = pipeline_unsupervised(X, n_clusters)
            cluster_results[n_clusters] = {"silhouette": silhouette, "cluster_labels": cluster_labels}
        feature_results[feature] = cluster_results
    fname = f"unsupervised_single_feature_all_sensors_{analysis_type}.pkl"
    save_results(feature_results, fname)
    logging.info("Done analysis 10!")


def run_analysis11(features, analysis_type):
    """Placeholder for analysis 11."""
    logging.info("TODO: Get best features from feature selection and run unsupervised learning")
    # Implement as needed


def main():
    parser = argparse.ArgumentParser(description="Run machine learning pipelines.")
    parser.add_argument("--analysis", type=int, default=1, help="Analysis number to run (1 to 11).")
    parser.add_argument("--analysis_type", type=str, default="all_ages", help="Analysis type (e.g., adolescent, child, all_ages).")
    args = parser.parse_args()
    
    analysis = args.analysis
    analysis_type = args.analysis_type
    
    features = load_and_preprocess_features(analysis_type)
    
    analysis_map = {
        1: run_analysis1,
        2: run_analysis2,
        3: run_analysis3,
        4: run_analysis4,
        5: run_analysis5,
        6: run_analysis6,
        7: run_analysis7,
        8: run_analysis8,
        9: run_analysis9,
        10: run_analysis10,
        11: run_analysis11,
    }
    
    if analysis in analysis_map:
        analysis_func = analysis_map[analysis]
        analysis_func(features, analysis_type)
    else:
        logging.error("Invalid analysis number. Choose from 1 to 11.")


if __name__ == "__main__":
    main()