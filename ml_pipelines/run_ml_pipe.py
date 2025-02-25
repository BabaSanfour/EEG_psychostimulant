import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config import data_dir, results_dir
import pickle
import pandas as pd
from ml_pipelines.pipelines import pipeline1_baseline, pipeline2_feature_selection, pipeline3_HP_search, pipeline4_feature_selection_HP_search









if __name__ == "__main__":
    features_file = os.path.join(data_dir, "csv", "features.csv")
    features = pd.read_csv(features_file)
    features = features.drop(columns=['dataset', 'id', 'age', 'sex', 'task', "subject"])
    features['group'] = features['group'].replace({'PAT': 1, 'CTR': 0})

    X = features.drop("group", axis=1)
    y = features["group"]
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

