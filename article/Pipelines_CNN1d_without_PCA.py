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

X = X.permute((0, 1, 2), (0, 2, 1))
y = y.permute((0, 1, 2), (0, 2, 1))
X_test = X_test.permute((0, 1, 2), (0, 2, 1))
y_test = y_test.permute((0, 1, 2), (0, 2, 1))

def scorer_rnmse(estimator, x, y):
    return rnmse(estimator.predict(x), y)

n_points = X.shape[0]
n_cv = 5
n_epochs = 50 #200

scores = {}


# In[14]:


import neuralop
from neuralop.models import TFNO


# In[29]:


class CNN(nn.Module):
    def __init__(self, input_dim=(X.shape[1:]), output_dim=(y.shape[1:])):
        super().__init__()

        self.features = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv1d(input_dim[0], input_dim[0]//4, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv1d(input_dim[0]//4, input_dim[0]//8, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv1d(input_dim[0]//8, input_dim[0]//4, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv1d(input_dim[0]//4, output_dim[0], kernel_size=3, padding='same'),
        )
        
    def forward(self, x):
        return self.features(x).squeeze()

device = 'cuda'

def get_rnmse():
    return rnmse

train = TensorDataset(X[:n_points], y[:n_points])
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

val_losses = []
k = 0
for fold, (train_idx, val_idx) in enumerate(kfold.split(train)):
    train_subset = DataLoader(Subset(train, train_idx), batch_size=128, shuffle=True)
    val_subset = DataLoader(Subset(train, val_idx), batch_size=128, shuffle=True)
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = rnmse
    model.train()
    for _ in (pbar:=tqdm(range(n_epochs))):
        for x_batch, y_batch in train_subset:
            optimizer.zero_grad()
            y_hat = model(x_batch.to(device))
            loss = loss_fn(y_hat, y_batch.to(device))
            loss.backward()
            pbar.set_description(f"Loss: {loss.item():.5f}")
            optimizer.step()
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        loss_val = 0
        for x_batch, y_batch in val_subset:
            y_hat = model(x_batch.to(device))
            loss_val += loss_fn(y_hat, y_batch.to(device))
        val_losses.append(loss_val.item()/len(val_subset))
        print(val_losses)
        torch.cuda.empty_cache()
        del model
    scores["CNN"] = val_losses

    pd.DataFrame(scores).to_csv(f'scores_CNN1d_without_PCA_{k}.csv')
    k += 1
