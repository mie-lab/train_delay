import pandas as pd
import numpy as np


def select_features(columns, version: str = "allfeatures") -> list:
    """Select a subset of the feature columns"""
    if version == "allfeatures":  # prev. version 2
        # All features
        use_features = [
            feat
            for feat in columns
            if feat.startswith("feat")
            # and not feat.startswith("feat_weather")
            # note: weather is not contained anymore in any case
        ]
    elif version == "markov_chain":  # prev. version 3
        print("Using features that are comparable features to Markov chain")
        feats_necessary = ["feat_delay_dep", "feat_obs_count", "feat_time_to_end_plan", "feat_avg_prev_delay"]
        # add delay in the past days
        hist_delay = [feat for feat in columns if feat.startswith("feat_delay_day")]
        # add historic final delay
        hist_final_delay = [feat for feat in columns if feat.startswith("feat_final_delay-day")]
        use_features = feats_necessary + hist_delay + hist_final_delay
        print(use_features)
    elif version == "submission":  # prev. version 4
        use_features = [
            "feat_delay_dep",
            "feat_obs_count",
            "feat_time_to_end_plan",
            "feat_avg_prev_delay",
            "feat_stops",
            "feat_delay_day-1",
            "feat_delay_day-2",
            "feat_delay_day-3",
            "feat_final_delay-day-1",
            "feat_final_delay-day-2",
            "feat_final_delay-day-3",
        ]
        print(use_features)
    elif version == "weather":
        use_features = [
            "feat_delay_dep",
            "feat_obs_count",
            "feat_time_to_end_plan",
            "feat_avg_prev_delay",
            "feat_stops",
            "feat_delay_day-1",
            "feat_delay_day-2",
            "feat_delay_day-3",
            "feat_final_delay-day-1",
            "feat_final_delay-day-2",
            "feat_final_delay-day-3",
            "feat_supplement_time",
            "feat_log_buffer_time",
        ] + [f for f in columns if "feat_weather" in f]
    elif version == "lasso":
        # see below for method to get these features
        use_features = [
            "feat_weather_rhum",
            "feat_supplement_time",
            "feat_delay_dep",
            "feat_delay_obs-1",
            "feat_delay_obs-5",
            "feat_avg_prev_delay",
            "feat_delay_on_day",
        ]
    else:
        raise NotImplementedError
    return use_features


def lasso_selection(data: pd.DataFrame):
    from sklearn.feature_selection import SelectFromModel
    from sklearn import linear_model

    feature_set = data.loc[data["use"] == "train", [c for c in data.columns if "feat" in c] + ["final_delay"]]
    X = feature_set.drop("final_delay", axis=1)
    y = feature_set["final_delay"].values
    # normalize
    X = (X - X.mean()) / X.std()

    print(X.shape, y.shape)

    # fit linear model
    clf = linear_model.Lasso(alpha=0.1).fit(X, y)
    # fit selection model
    model = SelectFromModel(clf, prefit=True, threshold=0.0001)

    return (np.array(X.columns)[model.get_support()]).tolist()
