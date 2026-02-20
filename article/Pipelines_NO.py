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
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Subset
from skorch import NeuralNet
from neuralop import FNO
import pandas as pd
import seaborn as sns
import matplotlib
sns.set_style('whitegrid')
sns.set_context('notebook')
sns.set_palette('hot', n_colors=7)
plt.rc('text', usetex=True)      

font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)

X = torch.cat([torch.load("../dataset/x_train.pt"), torch.load("../dataset/x_train_2.pt")])
y = torch.cat([torch.load("../dataset/y_train.pt"), torch.load("../dataset/y_train_2.pt")])
c = torch.cat([torch.load("../dataset/c_train.pt"), torch.load("../dataset/c_train_2.pt")])

X_test = torch.load("../dataset/x_test.pt")
y_test = torch.load("../dataset/y_test.pt")
c_test = torch.load("../dataset/c_test.pt")

class ResidualRegressor(BaseEstimator):
    def __init__(self, regressor, n_components=64):
        self.n_components = n_components
        self.pipe_lm = TransformedTargetRegressor(
            regressor=Pipeline([
                ("pca", PCA(n_components=self.n_components)),
                ("lm", LinearRegression(n_jobs=-1))
            ]),
            transformer=PCA(n_components=self.n_components),
            check_inverse=False
        )
        self.regressor = regressor

    def fit(self, X, y):
        X_np, y_np = X.numpy().reshape((X.shape[0], -1)), y.numpy().reshape((y.shape[0], -1))
        self.pipe_lm.fit(X_np, y_np)
        y_hat_lm = torch.tensor(self.pipe_lm.predict(X_np).reshape(y.shape))
        self.regressor.fit({'x': X, 'x_guess': y_hat_lm}, y)

    def predict(self, X):
        return torch.tensor(self.regressor.predict({'x': X, 'x_guess': torch.tensor(self.pipe_lm.predict(X.reshape((X.shape[0], -1))).reshape(X.shape))}))

def scorer_rnmse(estimator, x, y):
    return rnmse(estimator.predict(x), y)

n_points = 1024#X.shape[0]
n_cv = 5
n_epochs = 10#200

scores = {}


# In[14]:


import neuralop
from neuralop.models import TFNO


# In[29]:


class NO(nn.Module):
    def __init__(self, input_dim=(X.shape[1:]), output_dim=(y.shape[1:])):
        super().__init__()

        self.features = TFNO(
           n_modes=(32, 32),
           hidden_channels=16,
           in_channels=1,
           out_channels=1,
           factorization='tucker',
           implementation='factorized',
           rank=0.1
        )
        self.alpha, self.beta, self.gamma = nn.Parameter(torch.tensor([1/3])), nn.Parameter(torch.tensor([1/3])), nn.Parameter(torch.tensor([1/3]))

    def forward(self, x, x_guess):
        return self.alpha*x_guess + self.beta*x + self.gamma*self.features(x.unsqueeze(1)).squeeze()

device = 'mps'

def get_rnmse():
    return rnmse

train = TensorDataset(X[:n_points].to(device), y[:n_points].to(device))
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

val_losses = []
for fold, (train_idx, val_idx) in enumerate(kfold.split(train)):
    train_subset = DataLoader(Subset(train, train_idx), batch_size=128, shuffle=True)
    val_subset = DataLoader(Subset(train, val_idx), batch_size=128, shuffle=True)
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
    for x_batch, y_batch in train_subset:
        train_x_np.append(x_batch.cpu().detach().numpy().reshape(x_batch.shape[0], -1))
        train_y_np.append(y_batch.cpu().detach().numpy().reshape(y_batch.shape[0], -1))
    pipe_lm.fit(np.concatenate(train_x_np, axis=0), np.concatenate(train_y_np, axis=0))
    model = NO().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = rnmse
    model.train()
    for _ in (pbar:=tqdm(range(n_epochs))):
        for x_batch, y_batch in train_subset:
            optimizer.zero_grad()
            lm_pred = torch.tensor(pipe_lm.predict(x_batch.cpu().detach().numpy().reshape(x_batch.shape[0], -1))).to(device).reshape(x_batch.shape)
            y_hat = model(x_batch, lm_pred)
            loss = loss_fn(y_hat, y_batch)
            loss.backward()
            pbar.set_description(f"Loss: {loss.item():.5f}")
            optimizer.step()
    model.eval()
    loss_val = 0
    for x_batch, y_batch in val_subset:
        lm_pred = torch.tensor(pipe_lm.predict(x_batch.cpu().detach().numpy().reshape(x_batch.shape[0], -1))).to(device).reshape(x_batch.shape)
        y_hat = model(x_batch, lm_pred)
        loss_val += loss_fn(y_hat, y_batch)
    val_losses.append(loss_val.item()/len(val_subset))
    print(val_losses)
scores["NO"] = val_losses

pd.DataFrame(scores).to_csv('scores_no.csv')


# In[ ]:




