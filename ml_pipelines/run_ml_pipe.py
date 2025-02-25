import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config import data_dir, results_dir
import pickle
import pandas as pd
from ml_pipelines.pipelines import pipeline1_baseline, pipeline2_feature_selection, pipeline3_HP_search, pipeline4_feature_selection_HP_search
import argparse









if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run machine learning pipelines.")
    parser.add_argument("--analysis", type=int, default=1, help="Analysis number to run.")
    args = parser.parse_args()
    analysis = args.analysis
    features_file = os.path.join(data_dir, "csv", "features.csv")
    features = pd.read_csv(features_file)
    features = features.drop(columns=['dataset', 'id', 'age', 'sex', 'task', "subject"])
    features['group'] = features['group'].replace({'PAT': 1, 'CTR': 0})
    if analysis == 1:
        X = features.drop("group", axis=1)
        y = features["group"]
        # HP SEARCH FOR SINGLE FEATURE x SINGLE SENSOR
        model_results = {}
        for model_name in ["Decision Tree", "Random Forest", "Gradient Boosting", "K-Nearest Neighbors"]:
            print(f"Running pipeline for model: {model_name}")
            feature_results = {}
            for feature in X.columns:
                print(f"Running pipeline for feature: {feature}")
                acc = pipeline3_HP_search(X[[feature]], y, model_name)
                feature_results[feature] = acc
            model_results[model_name] = feature_results
        with open(os.path.join(results_dir, "single_feature_HP_results.pkl"), "wb") as f:
            pickle.dump(model_results, f)
        print("Done!")
    elif analysis == 2:
        # HP SEARCH FOR SINGLE FEATURE x ALL SENSORS
        feature_results = {}
        list_columns = features.columns
        list_features = [feat.split('.spaces-')[0] for feat in list_columns]
        clean_features = [feat.replace("feature-", "") for feat in list_features]
        for feature in clean_features:
            if feature == "group":
                continue
            select_columns = [col for col in list_columns if feature in col]
            X = features[select_columns]
            y = features["group"]
            print(f"Running classification for feature: {feature}")
            models_result = {}
            for model_name in ["Decision Tree", "Random Forest", "Gradient Boosting", "K-Nearest Neighbors"]:
                print(f"Running pipeline for model: {model_name}")
                results = pipeline3_HP_search(X, y, model_name)
                models_result[model_name] = results
            feature_results[feature] = models_result
        with open(os.path.join(results_dir, "single_feature_all_sensors_HP_results.pkl"), "wb") as f:
            pickle.dump(feature_results, f)
        print("Done!")
        

