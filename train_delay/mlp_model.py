import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from config import OUTLIER_CUTOFF

scaling_fun = {
    "sigmoid": (lambda x: torch.sigmoid(x) * 2 * OUTLIER_CUTOFF - OUTLIER_CUTOFF),  # scale from [0, 1] to [-5, 5]
    "tanh": (lambda x: torch.sigmoid(x) * OUTLIER_CUTOFF),  # scale from [-1, 1] to [-5, 5]
}


class TrainDelayMLP(nn.Module):
    def __init__(
        self,
        inp_size,
        out_size,
        dropout_rate=0.5,
        act="sigmoid",
        first_layer_size=128,
        second_layer_size=128,
        nr_layers=None,
        **kwargs,
    ):
        super(TrainDelayMLP, self).__init__()
        self.linear_1 = nn.Linear(inp_size, first_layer_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear_2 = nn.Linear(first_layer_size, second_layer_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.linear_3 = nn.Linear(second_layer_size, out_size)
        self.third_layer = nr_layers == 3
        # the None is used to ensure backward compatability (models that were not saved with linear_25 in state dict)
        if nr_layers is not None:
            self.linear_25 = nn.Linear(second_layer_size, second_layer_size)
        self.final_act = scaling_fun[act]
        print("initialized model with", first_layer_size, second_layer_size, nr_layers, dropout_rate)

    def forward(self, x):
        hidden = self.dropout1(torch.relu(self.linear_1(x)))
        hidden = self.dropout2(torch.relu(self.linear_2(hidden)))
        if self.third_layer:
            hidden = self.dropout2(torch.relu(self.linear_25(hidden)))
        out = self.final_act(self.linear_3(hidden))
        return out


class TrainDelayDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, ind):
        x = self.features[ind]
        y = self.labels[ind]
        return x, y


def attenuation_loss(output, y_true, validate=False):
    mu_pred, log_sigma_pred = output[:, 0], output[:, 1]

    # validation: only use mu
    if validate:
        return torch.mean((mu_pred - y_true) ** 2)

    sigma_pred = torch.exp(log_sigma_pred)  # undo the log
    return torch.mean(torch.log(sigma_pred**2) + ((y_true - mu_pred) / sigma_pred) ** 2)


def mse_loss(output, y_true, **kwargs):
    # assert output[:, 0].size() == y_true.size(), "Size mismatch!"
    return torch.mean((output[:, 0] - y_true) ** 2)


def average_batches(loss, n_average=1000):
    return np.array(loss)[: -(len(loss) % n_average)].reshape((-1, n_average)).mean(axis=1)


def train_model(
    model,
    train_set_nn_x,
    train_set_nn_y,
    val_set_nn_x,
    val_set_nn_y,
    criterion,
    batch_size=8,
    epochs=10,
    learning_rate=1e-5,
    save_path=os.path.join("test", "nn"),
    **kwargs,
):
    print(f"starting training with lr {learning_rate}")
    # create dataset and dataloader
    train_set_nn_torch = TrainDelayDataset(train_set_nn_x, train_set_nn_y)
    val_set_nn_torch = TrainDelayDataset(val_set_nn_x, val_set_nn_y)
    train_loader = DataLoader(train_set_nn_torch, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(val_set_nn_torch, batch_size=batch_size, shuffle=False)

    # init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    best_performance = np.inf
    epoch_test_loss, epoch_train_loss = [], []
    for epoch in range(epochs):
        losses = []
        for batch_num, input_data in enumerate(train_loader):
            optimizer.zero_grad()
            x, y = input_data
            x = x.to(device).float()
            y = y.to(device).float()

            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            if batch_num == 10:
                # print(model.linear_3.weight.grad)
                print("train loss at batch 10:", round(loss.item(), 2))
            if criterion == attenuation_loss:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            losses.append(loss.item())

            optimizer.step()

            # if batch_num == 10:
            #     print("\tEpoch %d | Batch %d | Loss %6.2f" % (epoch, batch_num, np.median(losses)))

        # TESTING
        model.eval()
        with torch.no_grad():
            test_losses = []
            for batch_num, input_data in enumerate(test_loader):
                x, y = input_data
                x = x.to(device).float()
                y = y.to(device)
                output = model(x)
                loss = criterion(output, y, validate=True)
                test_losses.append(loss.item())
        model.train()
        print(
            f"\n Epoch {epoch} | TRAIN Loss {sum(losses) / len(losses)} | TEST loss {sum(test_losses) / len(test_losses)} \n"
        )
        if sum(test_losses) < best_performance:
            best_performance = sum(test_losses)
            torch.save(model.state_dict(), os.path.join(save_path))
            print("Saved model")
        # print(
        #     f"\n Epoch {epoch} (median) | TRAIN Loss {round(np.median(losses), 3)} | TEST loss {round(np.median(test_losses), 3)} \n"
        # )
        epoch_test_loss.append(np.mean(test_losses))
        epoch_train_loss.extend(list(average_batches(losses)))
    plot_losses(epoch_train_loss, epoch_test_loss, save_path)


def plot_losses(losses, test_losses, name):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.subplot(1, 2, 2)
    plt.plot(test_losses)
    plt.tight_layout()
    plt.savefig(os.path.join(name + "_losses.png"))


def train_aleatoric(train_set_nn_x, train_set_nn_y, val_set_nn_x, val_set_nn_y, save_path="test", **kwargs):
    # init model with 2 outputs (mean and std)
    model = TrainDelayMLP(train_set_nn_x.shape[1], 2).to(device)
    criterion = attenuation_loss
    train_model(
        model,
        train_set_nn_x,
        train_set_nn_y,
        val_set_nn_x,
        val_set_nn_y,
        criterion,
        save_path=os.path.join(save_path, "nn_aleatoric"),
        **kwargs,
    )


def test_aleatoric(load_model, val_set_nn_x, **kwargs):
    model = TrainDelayMLP(val_set_nn_x.shape[1], 2).to(device)
    model.load_state_dict(torch.load(os.path.join(load_model, "nn_aleatoric")))
    # predict
    model.eval()
    pred = model(torch.from_numpy(val_set_nn_x).float())

    unc = pred[:, 1].detach().numpy()  # sigma
    pred = pred[:, 0].detach().numpy()  # mu

    return pred, np.exp(unc)


def train_test_time_dropout(
    train_set_nn_x, train_set_nn_y, val_set_nn_x, val_set_nn_y, dropout_rate=0.3, save_path="test", **kwargs
):
    model = TrainDelayMLP(train_set_nn_x.shape[1], 1, dropout_rate=dropout_rate).to(device)
    criterion = mse_loss

    train_model(
        model,
        train_set_nn_x,
        train_set_nn_y,
        val_set_nn_x,
        val_set_nn_y,
        criterion,
        save_path=os.path.join(save_path, "nn_dropout"),
        **kwargs,
    )


def test_test_time_dropout(load_model, val_set_nn_x, dropout_rate=0.5, **kwargs):
    model = TrainDelayMLP(val_set_nn_x.shape[1], 1, dropout_rate=dropout_rate).to(device)
    model.load_state_dict(torch.load(os.path.join(load_model, "nn_dropout")))
    model.train()  # Ensure that dropout is switched on

    # run for 10 times to get different predictions
    df_ttd = pd.DataFrame()
    for i in range(10):
        pred = model(torch.from_numpy(val_set_nn_x).float())
        df_ttd["run" + str(i)] = pred.detach().numpy().flatten()

    pred = np.mean(np.array(df_ttd), axis=1)
    unc = np.std(np.array(df_ttd), axis=1)
    return pred, unc


def train_unc_nn(train_set_nn_x, train_set_nn_y, val_set_nn_x, val_set_nn_y, save_path="test", **kwargs):
    """both aleatoric and epistemic uncertainty"""
    # init model with 2 outputs (mean and std)
    model = TrainDelayMLP(train_set_nn_x.shape[1], 2, **kwargs).to(device)
    # # to continue training
    # model.load_state_dict(torch.load(os.path.join(save_path, "nn_2")))
    criterion = attenuation_loss

    # save path that allows for training one nn per obs
    modified_save_path = os.path.join(
        save_path,
        f"nn-{kwargs['first_layer_size']}-{kwargs['second_layer_size']}-{kwargs['nr_layers']}-{kwargs['learning_rate']}-{kwargs['dropout_rate']}",
    )

    # train
    train_model(
        model,
        train_set_nn_x,
        train_set_nn_y,
        val_set_nn_x,
        val_set_nn_y,
        criterion,
        save_path=modified_save_path,
        **kwargs,
    )


def test_unc_nn(load_model, val_set_nn_x, nr_passes=10, **kwargs):
    """both aleatoric and epistemic uncertainty"""
    # init model with 2 outputs (mean and std)
    model = TrainDelayMLP(val_set_nn_x.shape[1], 2, **kwargs).to(device)
    # make flexible load_model path --> if nn already in path, don't add it
    if "nn" not in load_model:
        load_model = os.path.join(load_model, "nn")
    model.load_state_dict(torch.load(os.path.join(load_model)))
    model.train()  # Ensure that dropout is switched on

    # run for nr_passes times to get different predictions
    results = np.zeros((nr_passes, 2, len(val_set_nn_x)))
    for i in range(nr_passes):
        pred = model(torch.from_numpy(val_set_nn_x).float())
        results[i, 0, :] = pred[:, 0].detach().numpy().squeeze()
        results[i, 1, :] = np.exp(pred[:, 1].detach().numpy().squeeze())

    pred = np.mean(results[:, 0], axis=0)
    dropout_unc = np.std(results[:, 0], axis=0)
    aleatoric_unc = np.mean(results[:, 1], axis=0)

    return pred, dropout_unc + aleatoric_unc
    # # for aleatoric vs epistmic:
    # return pred, dropout_unc, aleatoric_unc
