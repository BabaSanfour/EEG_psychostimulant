import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config import data_dir, results_dir
import pickle
import pandas as pd
from ml_pipelines.pipelines import pipeline1_baseline, pipeline2_feature_selection, pipeline3_HP_search, pipeline4_feature_selection_HP_search, pipeline5_unsupervised
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
    elif analysis == 3:
        # Sensor selection per feature
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
                selected_features_dict = pipeline2_feature_selection(X, y, 20, model_name)
                models_result[model_name] = selected_features_dict
            feature_results[feature] = models_result
        with open(os.path.join(results_dir, "sensor_selection_per_feature.pkl"), "wb") as f:
            pickle.dump(feature_results, f)
        print("Done!")
    elif analysis == 4:
        # Sensor selection per feature then HP search
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
                selected_features_dict = pipeline4_feature_selection_HP_search(X, y, 0, 7, model_name)
                models_result[model_name] = selected_features_dict
            feature_results[feature] = models_result
        with open(os.path.join(results_dir, "sensor_selection_per_feature_HP_search.pkl"), "wb") as f:
            pickle.dump(feature_results, f)
        print("Done!")
    elif analysis == 5:
        # Baseline multifeatures all sensors
        X = features.drop("group", axis=1)
        y = features["group"]
        results, saved_models = pipeline1_baseline(X, y)
        combined_results = {"results": results, "models": saved_models}
        with open(os.path.join(results_dir, "baseline_results_all_feat_all_sensors.pkl"), "wb") as f:
            pickle.dump(combined_results, f)
    elif analysis == 6:
        # nultifeatures all sensors with HP search for all models
        X = features.drop("group", axis=1)
        y = features["group"]
        model_results = {}
        for model_name in ["Decision Tree", "Random Forest", "Gradient Boosting", "K-Nearest Neighbors"]:
            print(f"Running pipeline for model: {model_name}")
            results = pipeline3_HP_search(X, y, model_name)
            model_results[model_name] = results
        with open(os.path.join(results_dir, "HP_results_all_feat_all_sensors.pkl"), "wb") as f:
            pickle.dump(model_results, f)
        print("Done!")
    elif analysis == 7:
        # multifeature feature selection up to 40 features
        X = features.drop("group", axis=1)
        y = features["group"]
        models_result = {}
        for model_name in ["Decision Tree", "Random Forest", "Gradient Boosting", "K-Nearest Neighbors"]:
            print(f"Running pipeline for model: {model_name}")
            selected_features_dict = pipeline2_feature_selection(X, y, 40, model_name)
            models_result[model_name] = selected_features_dict
        with open(os.path.join(results_dir, "feature_selection_all_feat_all_sensors.pkl"), "wb") as f:
            pickle.dump(models_result, f)
        print("Done!")
    elif analysis == 8:
        # multifeature feature selection up to 40 features with HP search
        X = features.drop("group", axis=1)
        y = features["group"]
        models_result = {}
        for model_name in ["Decision Tree", "Random Forest", "Gradient Boosting", "K-Nearest Neighbors"]:
            print(f"Running pipeline for model: {model_name}")
            selected_features_dict = pipeline4_feature_selection_HP_search(X, y, 0, 40, model_name)
            models_result[model_name] = selected_features_dict
        with open(os.path.join(results_dir, "feature_selection_HP_search_all_feat_all_sensors.pkl"), "wb") as f:
            pickle.dump(models_result, f)
        print("Done!")
    elif analysis == 9:
        # unsupervised learning, all feature all sensors
        X = features.drop("group", axis=1)
        clusters_results = {}
        for n_clusters in range(2, 11):
            print(f"Running pipeline for {n_clusters} clusters")
            silhouette, cluster_labels = pipeline5_unsupervised(X, n_clusters)
            results = {"silhouette": silhouette, "cluster_labels": cluster_labels}
            clusters_results[n_clusters] = results
        with open(os.path.join(results_dir, "unsupervised_single_feature_all_sensors.pkl"), "wb") as f:
            pickle.dump(clusters_results, f)
        print("Done!")
    elif analysis == 10:
        feature_results = {}
        list_columns = features.columns
        list_features = [feat.split('.spaces-')[0] for feat in list_columns]
        clean_features = [feat.replace("feature-", "") for feat in list_features]
        for feature in clean_features:
            if feature == "group":
                continue
            select_columns = [col for col in list_columns if feature in col]
            X = features[select_columns]
            cluster_results = {}
            for n_clusters in range(2, 11):
                print(f"Running pipeline for {n_clusters} clusters")
                silhouette, cluster_labels = pipeline5_unsupervised(X, n_clusters)
                results = {"silhouette": silhouette, "cluster_labels": cluster_labels}
                cluster_results[n_clusters] = results
            feature_results[feature] = cluster_results
        with open(os.path.join(results_dir, "unsupervised_single_feature_all_sensors.pkl"), "wb") as f:
            pickle.dump(feature_results, f)
        print("Done!")
    elif analysis == 11:
        print("TODO: Get best features from feature selection and run unsupervised learning")
    else:    
        print("Invalid analysis number.")
        print("Choose from 1 to 11.")
    # TODO: re run all analysis with 2 age groups
# Run the script with the following command:
# python run_ml_pipe.py --analysis 1