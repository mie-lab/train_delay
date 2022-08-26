import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pandas as pd

device = "cpu"
from config import OUTLIER_CUTOFF

scaling_fun = {
    "sigmoid": (lambda x: torch.sigmoid(x) * 2 * OUTLIER_CUTOFF - OUTLIER_CUTOFF),  # scale from [0, 1] to [-5, 5]
    "tanh": (lambda x: torch.sigmoid(x) * OUTLIER_CUTOFF),  # scale from [-1, 1] to [-5, 5]
}


class TrainDelayMLP(nn.Module):
    def __init__(self, inp_size, out_size, dropout_rate=0, act="sigmoid"):
        super(TrainDelayMLP, self).__init__()
        self.linear_1 = nn.Linear(inp_size, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear_2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.linear_3 = nn.Linear(128, out_size)
        self.final_act = scaling_fun[
            act
        ]  # TODO: aleatoric --> activation function should only be for mean, not for std!!

    def forward(self, x):
        hidden = self.dropout1(torch.relu(self.linear_1(x)))
        hidden = self.dropout2(torch.relu(self.linear_2(hidden)))
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
    return torch.mean(torch.log(sigma_pred ** 2) + ((y_true - mu_pred) / sigma_pred) ** 2)


def mse_loss(output, y_true, **kwargs):
    # assert output[:, 0].size() == y_true.size(), "Size mismatch!"
    return torch.mean((output[:, 0] - y_true) ** 2)


def average_batches(loss, n_average=1000):
    return np.array(loss)[: -(len(loss) % n_average)].reshape((-1, n_average)).mean(axis=1)


def train_model(model, epochs, train_loader, test_loader, criterion, name="nn"):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

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
            torch.save(model.state_dict(), os.path.join("trained_models", name))
            print("Saved model")
        # print(
        #     f"\n Epoch {epoch} (median) | TRAIN Loss {round(np.median(losses), 3)} | TEST loss {round(np.median(test_losses), 3)} \n"
        # )
        epoch_test_loss.append(np.mean(test_losses))
        epoch_train_loss.extend(list(average_batches(losses)))
    plot_losses(epoch_train_loss, epoch_test_loss, name)


def plot_losses(losses, test_losses, name):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.subplot(1, 2, 2)
    plt.plot(test_losses)
    plt.tight_layout()
    plt.savefig(os.path.join("trained_models", name + "_losses.png"))


def fit_mlp_aleatoric(train_set_nn_x, train_set_nn_y, val_set_nn_x, val_set_nn_y, epochs=1, load_model=None, **kwargs):
    model = TrainDelayMLP(train_set_nn_x.shape[1], 2)

    if load_model is None:
        criterion = attenuation_loss
        batch_size = 8
        train_set_nn_torch = TrainDelayDataset(train_set_nn_x, train_set_nn_y)
        val_set_nn_torch = TrainDelayDataset(val_set_nn_x, val_set_nn_y)
        train_loader = DataLoader(train_set_nn_torch, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(val_set_nn_torch, batch_size=batch_size, shuffle=False)
        train_model(model, epochs, train_loader, test_loader, criterion, name="nn_aleatoric")
    else:
        model.load_state_dict(torch.load(os.path.join("trained_models", load_model, "nn_aleatoric")))

    # predict
    model.eval()
    pred = model(torch.from_numpy(val_set_nn_x).float())

    unc = pred[:, 1].detach().numpy()  # sigma
    pred = pred[:, 0].detach().numpy()  # mu

    return pred, np.exp(unc)


def fit_mlp_test_time_dropout(
    train_set_nn_x, train_set_nn_y, val_set_nn_x, val_set_nn_y, epochs=1, dropout_rate=0.3, load_model=None, **kwargs
):

    model = TrainDelayMLP(train_set_nn_x.shape[1], 1, dropout_rate=dropout_rate)

    if load_model is None:
        criterion = mse_loss
        batch_size = 8
        train_set_nn_torch = TrainDelayDataset(train_set_nn_x, train_set_nn_y)
        val_set_nn_torch = TrainDelayDataset(val_set_nn_x, val_set_nn_y)
        train_loader = DataLoader(train_set_nn_torch, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(val_set_nn_torch, batch_size=batch_size, shuffle=False)
        train_model(model, epochs, train_loader, test_loader, criterion, name="nn_dropout")
    else:
        model.load_state_dict(torch.load(os.path.join("trained_models", load_model, "nn_dropout")))
        model.train()  # Ensure that dropout is switched on
    df_ttd = pd.DataFrame()
    for i in range(10):
        pred = model(torch.from_numpy(val_set_nn_x).float())
        df_ttd["run" + str(i)] = pred.detach().numpy().flatten()

    pred = np.mean(np.array(df_ttd), axis=1)
    unc = np.std(np.array(df_ttd), axis=1)

    return pred, unc
