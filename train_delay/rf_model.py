from sklearn.ensemble import RandomForestRegressor
import numpy as np

def fit_rf_model(train_set_rf_x, train_set_rf_y, val_set_rf_x, val_set_rf_y, use_features=None):
    # init tree
    regr = RandomForestRegressor(max_depth=30, random_state=0)
    # fit
    regr.fit(train_set_rf_x, train_set_rf_y)

    # predict
    rf_pred = regr.predict(val_set_rf_x)

    # get prediction per tree
    per_tree_pred = [tree.predict(val_set_rf_x) for tree in regr.estimators_]
    rf_unc = np.std(per_tree_pred, axis=0)

    if use_features is not None:
        print("Most important features:")
        print(np.array(use_features)[np.argsort(regr.feature_importances_)][::-1])

    return rf_pred, rf_unc