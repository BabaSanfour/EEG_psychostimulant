import os
import sys
import pickle
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config import data_dir, results_dir

def pipeline1_baseline(X, y, use_preprocessing=False):
    """
    Baseline pipeline.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    saved_models = {}
    for model_name, model in models.items():
        if use_preprocessing:
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            steps = [("scaler", StandardScaler())]
            X_processed = Pipeline(steps).fit_transform(X)
        else:
            X_processed = X.values

        scores = cross_val_score(model, X_processed, y, cv=cv, scoring="accuracy")
        score_mean = scores.mean()
        print(f"Accuracy: {score_mean:.4f}")
        results[model_name] = score_mean
        model.fit(X_processed, y)
        saved_models[model_name] = model
    # try:
    #     final_output_file = os.path.join(results_dir, "baseline_results.pkl")
    #     with open(final_output_file, "wb") as f:
    #         pickle.dump(results, f)
    #     print(f"Final results saved to {final_output_file}")
    # except Exception as e:
    #     print(f"Error saving final results: {e}")
    return results, saved_models

def pipeline2_feature_selection(X, y, num_features, model_name, use_preprocessing=False):
    """
    Feature selection pipeline.
    """
    from sklearn.feature_selection import SequentialFeatureSelector
    base_model = models.get(model_name, None)
    if base_model is None:
        raise ValueError(f"Model '{model_name}' is not defined in the models dictionary.")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    if use_preprocessing:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        steps = [("scaler", StandardScaler())]
        X_processed = Pipeline(steps).fit_transform(X)
    else:
        X_processed = X.values
    feature_counts = list(range(1, num_features + 1))
    selected_features_dict = {}
    for k in feature_counts:
        print(f"Processing {k} features")
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
        print(f"Accuracy with {k} features: {score_mean:.4f}")
        result_dict = {
            "selected_features": selected_features,
            "accuracy": score_mean,
            "fitted_model": sfs.estimator
        }
        if hasattr(sfs.estimator, "feature_importances_"):
            result_dict["feature_importances"] = sfs.estimator.feature_importances_
        selected_features_dict[k] = result_dict

        # Save intermediate results every 10 features
        if k % 10 == 0:
            output_file = os.path.join(results_dir, f"{model_name}_selected_features_{k}.pkl")
            try:
                with open(output_file, "wb") as f:
                    pickle.dump(selected_features_dict, f)
                print(f"Intermediate results saved to {output_file}")
            except Exception as e:
                print(f"Error saving intermediate results for {k} features: {e}")

    # Save final results
    final_output_file = os.path.join(results_dir, f"{model_name}_selected_features_final.pkl")
    try:
        with open(final_output_file, "wb") as f:
            pickle.dump(selected_features_dict, f)
        print(f"Final results saved to {final_output_file}")
    except Exception as e:
        print(f"Error saving final results: {e}")

def pipeline3_HP_search(X, y, model_name, use_preprocessing=False):
    """
    Hyperparameter search pipeline.
    """
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    base_model = models.get(model_name, None)
    if base_model is None:
        raise ValueError(f"Model '{model_name}' is not defined in the models dictionary.")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    if use_preprocessing:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        steps = [("scaler", StandardScaler())]
        X_processed = Pipeline(steps).fit_transform(X)
    else:
        X_processed = X.values
    param_distributions = params_grids.get(model_name, None)
    search = GridSearchCV(
        base_model,
        param_grid=param_distributions,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )
    search.fit(X_processed, y)
    print(f"Best parameters: {search.best_params_}")
    print(f"Best accuracy: {search.best_score_:.4f}")
    # try:
    #     final_output_file = os.path.join(results_dir, f"{model_name}_hyperparameter_search.pkl")
    #     with open(final_output_file, "wb") as f:
    #         pickle.dump(search.cv_results_, f)
    #     print(f"Final results saved to {final_output_file}")
    # except Exception as e:
    #     print(f"Error saving final results: {e}")
    return search.best_score_

def pipeline4_feature_selection_HP_search(X, y, start_feature, num_features, model_name, use_preprocessing=False):
    """
    Feature selection and hyperparameter search pipeline.
    """
    from sklearn.feature_selection import SequentialFeatureSelector
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    base_model = models.get(model_name, None)
    if base_model is None:
        raise ValueError(f"Model '{model_name}' is not defined in the models dictionary.")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    if use_preprocessing:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        steps = [("scaler", StandardScaler())]
        X_processed = Pipeline(steps).fit_transform(X)
    else:
        X_processed = X.values
    feature_counts = list(range(start_feature, num_features + 1))
    selected_features_dict = {}
    for k in feature_counts:
        print(f"Processing {k} features")
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
        param_distributions = params_grids.get(model_name, None)
        base_model = models.get(model_name, None)
        search = GridSearchCV(
            base_model,
            param_grid=param_distributions,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )
        search.fit(X_selected, y)
        print(f"Best parameters with {k} features: {search.best_params_}")
        print(f"Best accuracy with {k} features: {search.best_score_:.4f}")
        result_dict = {
            "selected_features": selected_features,
            "accuracy": search.best_score_,
            "best_params": search.best_params_,
            "fitted_model": search.best_estimator_
        }
        if hasattr(search.best_estimator_, "feature_importances_"):
            result_dict["feature_importances"] = search.best_estimator_.feature_importances_
        selected_features_dict[k] = result_dict

        # Save intermediate results every 10 features
        if k % 10 == 0:
            output_file = os.path.join(results_dir, f"{model_name}_selected_features_HP_search_{k}.pkl")
            try:
                with open(output_file, "wb") as f:
                    pickle.dump(selected_features_dict, f)
                print(f"Intermediate results saved to {output_file}")
            except Exception as e:
                print(f"Error saving intermediate results for {k} features: {e}")