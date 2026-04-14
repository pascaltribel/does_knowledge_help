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
from neuralop import FNO
import pandas as pd
import seaborn as sns
import matplotlib
from time import time
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


# In[2]:

print("Reading files...")
x = torch.cat([torch.load("../dataset/x_train.pt"), torch.load("../dataset/x_train_2.pt")])
y = torch.cat([torch.load("../dataset/y_train.pt"), torch.load("../dataset/y_train_2.pt")])
c = torch.cat([torch.load("../dataset/c_train.pt"), torch.load("../dataset/c_train_2.pt")])

x_test = torch.load("../dataset/x_test.pt")
y_test = torch.load("../dataset/y_test.pt")
c_test = torch.load("../dataset/c_test.pt")


# In[3]:

n_points = 1024#x.shape[0]
n_cv = 5
scores = {}

# In[4]:


x_np, y_np, c_np = x.numpy(), y.numpy(), c.numpy()
x_np_reshaped, y_np_reshaped, c_np_reshaped = x_np.reshape((x_np.shape[0], -1)), y_np.reshape((y_np.shape[0], -1)), c_np.reshape((c_np.shape[0], -1))
x_test_np, y_test_np, c_test_np = x_test.numpy(), y_test.numpy(), c_test.numpy()
x_test_np_reshaped, y_test_np_reshaped, c_test_np_reshaped = x_test_np.reshape((x_test_np.shape[0], -1)), y_test_np.reshape((y_test_np.shape[0], -1)), c_test_np.reshape((c_test_np.shape[0], -1))


# In[5]:


class DoNothingRegressor(BaseEstimator):
    def __init__(self):
        pass
    def fit(self, X, y):
        pass
    def predict(self, X):
        return np.zeros_like(X)


class NoResidualRegressor(BaseEstimator):
    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, X, y):
        self.regressor.fit(X, y)

    def predict(self, X):
        return self.regressor.predict(X)

def scorer_rnmse(estimator, x, y):
    return rnmse(estimator.predict(x), y)

t = time()
print("LM...")

scores["LM"] = cross_val_score(
    NoResidualRegressor(regressor=LinearRegression(n_jobs=-1)),
    x_np_reshaped[:n_points], y_np_reshaped[:n_points],
    cv=n_cv,
    scoring=scorer_rnmse,
    verbose=1
)
print(f"Done in {time()-t}s")


# In[13]:

t = time()
print("KNN...")

from sklearn.neighbors import KNeighborsRegressor
scores["KNN"] = cross_val_score(
    NoResidualRegressor(regressor=KNeighborsRegressor(10)),
    x_np_reshaped[:n_points], y_np_reshaped[:n_points],
    cv=n_cv,
    scoring=scorer_rnmse,
    verbose=1
)
print(f"Done in {time()-t}s")

t = time()
print("MLP...")

from sklearn.neural_network import MLPRegressor

scores["MLP"] = cross_val_score(
    NoResidualRegressor(regressor=MLPRegressor(max_iter=1000)),
    x_np_reshaped[:n_points], y_np_reshaped[:n_points],
    cv=n_cv,
    scoring=scorer_rnmse,
    verbose=1
)

print(f"Done in {time()-t}s")

# In[17]:

t = time()
print("RF...")

from sklearn.ensemble import RandomForestRegressor
scores["RF"] = cross_val_score(
    NoResidualRegressor(regressor=RandomForestRegressor(n_jobs=-1)),
    x_np_reshaped[:n_points], y_np_reshaped[:n_points],
    cv=n_cv,
    scoring=scorer_rnmse,
    verbose=1
)

print(f"Done in {time()-t}s")


print("Saving data...")

data = pd.DataFrame(scores)[sorted(scores, key=lambda x: np.mean(scores[x]), reverse=True)]
data.to_csv('results_without_pcr.csv')


print("Done.")
