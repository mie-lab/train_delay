import sklearn.metrics as me


def get_metrics(pred, name="model"):
    pred = pred.dropna()
    x = pred["pred"].values
    y = pred["final_delay"].values
    res_dict = {name: {"MSE": me.mean_squared_error(x, y), "MAE": me.mean_absolute_error(x, y)}}
    return res_dict
