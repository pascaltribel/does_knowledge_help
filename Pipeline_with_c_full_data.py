#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from utils import rnmse
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.compose import TransformedTargetRegressor
from torch.utils.data import Subset
from neuralop import FNO
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('notebook')
sns.set_palette('hot', n_colors=7)


# In[2]:


x = torch.cat([torch.load("dataset/x_train.pt"), torch.load("dataset/x_train_2.pt")])
y = torch.cat([torch.load("dataset/y_train.pt"), torch.load("dataset/y_train_2.pt")])
c = torch.cat([torch.load("dataset/c_train.pt"), torch.load("dataset/c_train_2.pt")])

x_test = torch.load("dataset/x_test.pt")
y_test = torch.load("dataset/y_test.pt")
c_test = torch.load("dataset/c_test.pt")


# In[3]:


x0, y0, c0 = x[0], y[0], c[0]


# In[4]:


print(x0.shape, y0.shape, c0.shape)


# In[5]:


class PermuteLayer(torch.nn.Module):
    dims: tuple[int, ...]

    def __init__(self, dims: tuple[int, ...]) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.permute(*self.dims)
        
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_x = nn.Sequential(
            nn.Linear(x0.shape[-1], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            PermuteLayer((0, 2, 1)),
            nn.Linear(256, 128),
            nn.ReLU(),
            PermuteLayer((0, 2, 1)),
        )
        self.net_c = nn.Sequential(
            nn.Linear(c0.shape[-1], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.net_y = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, y0.shape[-1])
        )

    def forward(self, x, c):
        return self.net_y(torch.concatenate([self.net_x(x), self.net_c(c)], axis=1))


# In[6]:


def scorer_rnmse(estimator, x, y):
    return rnmse(estimator.predict(x), y)

def get_rnmse():
    return rnmse


# In[7]:


device = 'mps'
n_points = x.shape[0]
n_epochs = 8
scores = {}


# In[8]:


n_components = 128

x_np, y_np, c_np = x.numpy(), y.numpy(), c.numpy()
x_np_reshaped, y_np_reshaped, c_np_reshaped = x_np.reshape((x_np.shape[0], -1)), y_np.reshape((y_np.shape[0], -1)), c_np.reshape((c_np.shape[0], -1))
x_test_np, y_test_np, c_test_np = x_test.numpy(), y_test.numpy(), c_test.numpy()
x_test_np_reshaped, y_test_np_reshaped, c_test_np_reshaped = x_test_np.reshape((x_test_np.shape[0], -1)), y_test_np.reshape((y_test_np.shape[0], -1)), c_test_np.reshape((c_test_np.shape[0], -1))


# In[17]:


TRAIN_MLP, TRAIN_CNN, TRAIN_FNO = True, True, True


# In[10]:


pipe_lm = TransformedTargetRegressor(
    regressor=Pipeline([
        ("pca", PCA(n_components=n_components)),
        ("lm", LinearRegression(n_jobs=-1))
    ]),
    transformer=PCA(n_components=n_components),
    check_inverse=False
)

scores["LM"] = cross_val_score(
    pipe_lm,
    x_np_reshaped[:n_points], y_np_reshaped[:n_points],
    cv=5,
    scoring=scorer_rnmse
)


# In[11]:


if TRAIN_MLP:
    train = TensorDataset(x[:n_points].to(device), c[:n_points].to(device), y[:n_points].to(device))
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    val_losses = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train)):
        train_subset = DataLoader(Subset(train, train_idx), batch_size=32, shuffle=True)
        val_subset = DataLoader(Subset(train, val_idx), batch_size=256, shuffle=True)
        pipe_lm = TransformedTargetRegressor(
            regressor=Pipeline([
                ("pca", PCA(n_components=512)),
                ("lm", LinearRegression(n_jobs=-1))
            ]),
            transformer=PCA(n_components=512),
            check_inverse=False
        )
        train_x_np = []
        train_y_np = []
        for x_batch, c_batch, y_batch in train_subset:
            train_x_np.append(x_batch.cpu().detach().numpy().reshape(x_batch.shape[0], -1))
            train_y_np.append(y_batch.cpu().detach().numpy().reshape(y_batch.shape[0], -1))
        pipe_lm.fit(np.concatenate(train_x_np, axis=0), np.concatenate(train_y_np, axis=0))
        model = MLP().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = rnmse
        model.train()
        for _ in (pbar:=tqdm(range(n_epochs))):
            for x_batch, c_batch, y_batch in train_subset:
                optimizer.zero_grad()
                lm_pred = torch.tensor(pipe_lm.predict(x_batch.cpu().detach().numpy().reshape(x_batch.shape[0], -1))).to(device).reshape(x_batch.shape)
                y_hat = lm_pred + model(x_batch, c_batch)
                loss = loss_fn(y_hat, y_batch)
                loss.backward()
                pbar.set_description(f"Loss: {loss.item():.5f}")
                optimizer.step()
        model.eval()
        loss_val = 0
        for x_batch, c_batch, y_batch in val_subset:
            lm_pred = torch.tensor(pipe_lm.predict(x_batch.cpu().detach().numpy().reshape(x_batch.shape[0], -1))).to(device).reshape(x_batch.shape)
            y_hat = lm_pred + model(x_batch, c_batch)
            loss_val += loss_fn(y_hat, y_batch)
        val_losses.append(loss_val.item()/len(val_subset))
        print(val_losses)
    
    scores["MLP"] = val_losses


# In[18]:


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_x = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding="same"),
            nn.Tanh(),
            nn.Conv2d(4, 8, kernel_size=3, padding="same"),
            nn.Tanh(),
            nn.MaxPool2d((2, 1)),
        )
        self.net_c = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding="same"),
            nn.Tanh(),
            nn.Conv2d(4, 8, kernel_size=3, padding="same"),
            nn.Tanh(),
        )
        self.net_y = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding="same"),
            nn.Tanh(),
            nn.Conv2d(8, 1, kernel_size=3, padding="same"),
            nn.Tanh(),
            nn.ConvTranspose2d(1, 1, kernel_size=(2, 1), stride=(2, 1))
        )

    def forward(self, x, c):
        x, c = x.unsqueeze(1), c.unsqueeze(1)
        return self.net_y(torch.concatenate([self.net_x(x), self.net_c(c)], axis=1)).squeeze()


# In[ ]:


device = 'cpu'
if TRAIN_CNN:
    train = TensorDataset(x[:n_points].to(device), c[:n_points].to(device), y[:n_points].to(device))
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    val_losses = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train)):
        train_subset = DataLoader(Subset(train, train_idx), batch_size=32, shuffle=True)
        val_subset = DataLoader(Subset(train, val_idx), batch_size=32, shuffle=True)
        pipe_lm = TransformedTargetRegressor(
            regressor=Pipeline([
                ("pca", PCA(n_components=512)),
                ("lm", LinearRegression(n_jobs=-1))
            ]),
            transformer=PCA(n_components=512),
            check_inverse=False
        )
        train_x_np = []
        train_y_np = []
        for x_batch, c_batch, y_batch in train_subset:
            train_x_np.append(x_batch.cpu().detach().numpy().reshape(x_batch.shape[0], -1))
            train_y_np.append(y_batch.cpu().detach().numpy().reshape(y_batch.shape[0], -1))
        pipe_lm.fit(np.concatenate(train_x_np, axis=0), np.concatenate(train_y_np, axis=0))
        model = CNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = rnmse
        model.train()
        for _ in (pbar:=tqdm(range(n_epochs))):
            for x_batch, c_batch, y_batch in train_subset:
                optimizer.zero_grad()
                lm_pred = torch.tensor(pipe_lm.predict(x_batch.cpu().detach().numpy().reshape(x_batch.shape[0], -1))).to(device).reshape(x_batch.shape)
                y_hat = lm_pred + model(x_batch, c_batch)
                loss = loss_fn(y_hat, y_batch)
                loss.backward()
                pbar.set_description(f"Loss: {loss.item():.5f}")
                optimizer.step()
        model.eval()
        loss_val = 0
        for x_batch, c_batch, y_batch in val_subset:
            lm_pred = torch.tensor(pipe_lm.predict(x_batch.cpu().detach().numpy().reshape(x_batch.shape[0], -1))).to(device).reshape(x_batch.shape)
            y_hat = lm_pred + model(x_batch, c_batch)
            loss_val += loss_fn(y_hat, y_batch)
        val_losses.append(loss_val.item()/len(val_subset))
        print(val_losses)
    scores["CNN"] = val_losses


# In[ ]:


class FNOp(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_x = nn.Sequential(
            nn.MaxPool2d((2, 1)),
        )
        self.fno = nn.Sequential(
            FNO(n_modes=(16, 16), hidden_channels=16, in_channels=2, out_channels=1),
        )
        self.tail = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=(2, 1), stride=(2, 1))
        )

    def forward(self, x, c):
        x, c = x.unsqueeze(1), c.unsqueeze(1)
        return self.tail(self.fno(torch.concatenate([self.net_x(x), c], axis=1))).squeeze()


# In[ ]:


if TRAIN_FNO:
    train = TensorDataset(x[:n_points].to(device), c[:n_points].to(device), y[:n_points].to(device))
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    val_losses = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train)):
        train_subset = DataLoader(Subset(train, train_idx), batch_size=32, shuffle=True)
        val_subset = DataLoader(Subset(train, val_idx), batch_size=32, shuffle=True)
        pipe_lm = TransformedTargetRegressor(
            regressor=Pipeline([
                ("pca", PCA(n_components=512)),
                ("lm", LinearRegression(n_jobs=-1))
            ]),
            transformer=PCA(n_components=512),
            check_inverse=False
        )
        train_x_np = []
        train_y_np = []
        for x_batch, c_batch, y_batch in train_subset:
            train_x_np.append(x_batch.cpu().detach().numpy().reshape(x_batch.shape[0], -1))
            train_y_np.append(y_batch.cpu().detach().numpy().reshape(y_batch.shape[0], -1))
        pipe_lm.fit(np.concatenate(train_x_np, axis=0), np.concatenate(train_y_np, axis=0))
        model = FNOp().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = rnmse
        model.train()
        for _ in (pbar:=tqdm(range(n_epochs))):
            for x_batch, c_batch, y_batch in train_subset:
                optimizer.zero_grad()
                lm_pred = torch.tensor(pipe_lm.predict(x_batch.cpu().detach().numpy().reshape(x_batch.shape[0], -1))).to(device).reshape(x_batch.shape)
                y_hat = lm_pred + model(x_batch, c_batch)
                loss = loss_fn(y_hat, y_batch)
                loss.backward()
                pbar.set_description(f"Loss: {loss.item():.5f}")
                optimizer.step()
        model.eval()
        loss_val = 0
        for x_batch, c_batch, y_batch in val_subset:
            lm_pred = torch.tensor(pipe_lm.predict(x_batch.cpu().detach().numpy().reshape(x_batch.shape[0], -1))).to(device).reshape(x_batch.shape)
            y_hat = lm_pred + model(x_batch, c_batch)
            loss_val += loss_fn(y_hat, y_batch)
        val_losses.append(loss_val.item()/len(val_subset))
        print(val_losses)
    scores["FNO"] = val_losses


# In[ ]:


plt.figure(figsize=(7, 7))
sns.boxplot(pd.DataFrame(scores))
plt.grid(True)
plt.ylabel("5-fold CV RNMSE")
plt.xticks(rotation=45)
plt.savefig("5cvrnmse.jpg", dpi=150)
plt.show()


# In[ ]:




