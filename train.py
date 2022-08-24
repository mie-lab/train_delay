import os
import pandas as pd
import argparse

from run import split_train_test, get_train_val_test, MODEL_FUNC_DICT

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inp_path", type=str, default=os.path.join("data", "data_enriched.csv"))
    parser.add_argument("-m", "--method", default="rf", type=str, help="one of rf, nn_aleatoric or nn_dropout")
    parser.add_argument("-e", "--epochs", default=1, type=int, help="number of epochs")
    args = parser.parse_args()

    epochs = args.epochs
    model_func = MODEL_FUNC_DICT[args.method]

    data = pd.read_csv(args.inp_path)
    data.index.name = "id"

    # split into train, val and test
    train_set, val_set, test_set = split_train_test(data)

    use_features = ["delay_dep", "obs_count"] + [feat for feat in data.columns if feat.startswith("feat")]

    (train_set_nn_x, train_set_nn_y, val_set_nn_x, val_set_nn_y) = get_train_val_test(
        train_set, val_set, test_set, use_features, training=True
    )
    print("DATA SHAPES", train_set_nn_x.shape, train_set_nn_y.shape, val_set_nn_x.shape, val_set_nn_y.shape)

    model_func(
        train_set_nn_x,
        train_set_nn_y,
        val_set_nn_x,
        val_set_nn_y,
        epochs=epochs,
        dropout_rate=0.5,
        load_model=None,
        use_features=use_features,
    )
