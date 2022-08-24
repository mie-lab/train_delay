import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pandas as pd

device = "cpu"


class TrainDelayMLP(nn.Module):
    def __init__(self, inp_size, out_size, dropout_rate=0):
        super(TrainDelayMLP, self).__init__()
        self.linear_1 = nn.Linear(inp_size, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear_2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.linear_3 = nn.Linear(128, out_size)

    def forward(self, x):
        hidden = self.dropout1(torch.relu(self.linear_1(x)))
        hidden = self.dropout2(torch.relu(self.linear_2(hidden)))
        out = self.linear_3(hidden)
        return out  # no activation function


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
    return torch.mean((output - y_true) ** 2)


def train_model(model, epochs, train_loader, test_loader, criterion):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    model.train()
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
            if criterion == attenuation_loss:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            losses.append(loss.item())

            optimizer.step()

            if batch_num == 10:
                print("\tEpoch %d | Batch %d | Loss %6.2f" % (epoch, batch_num, np.median(losses)))

        with torch.no_grad():
            test_losses = []
            for batch_num, input_data in enumerate(test_loader):
                x, y = input_data
                x = x.to(device).float()
                y = y.to(device)
                output = model(x)
                loss = criterion(output, y, validate=True)
                if batch_num == 10:
                    print(loss.item())
                test_losses.append(loss.item())

        #         print(
        #             f"\n Epoch {epoch} | TRAIN Loss {sum(losses) / len(losses)} | TEST loss {sum(test_losses) / len(test_losses)} \n"
        #         )
        print(
            f"\n Epoch {epoch} (median) | TRAIN Loss {round(np.median(losses), 3)} | TEST loss {round(np.median(test_losses), 3)} \n"
        )
    return losses, test_losses


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
        losses, test_losses = train_model(model, epochs, train_loader, test_loader, criterion)

        plot_losses(losses, test_losses, "aleatoric_nn")

        torch.save(model.state_dict(), os.path.join("trained_models", "aleatoric_nn"))
    else:
        model.load_state_dict(torch.load(os.path.join("trained_models", load_model, "aleatoric_nn")))

    # predict
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
        losses, test_losses = train_model(model, epochs, train_loader, test_loader, criterion)
        plot_losses(losses, test_losses, "dropout_nn")
        torch.save(model.state_dict(), os.path.join("trained_models", "dropout_nn"))
    else:
        model.load_state_dict(torch.load(os.path.join("trained_models", load_model, "dropout_nn")))

    df_ttd = pd.DataFrame()
    for i in range(10):
        pred = model(torch.from_numpy(val_set_nn_x).float())
        df_ttd["run" + str(i)] = pred.detach().numpy().flatten()

    pred = np.mean(np.array(df_ttd), axis=1)
    unc = np.std(np.array(df_ttd), axis=1)

    return pred, unc
