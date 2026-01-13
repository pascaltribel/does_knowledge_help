import torch
from torch.utils.data import DataLoader, TensorDataset
import math
import numpy as np


def ricker(dt, pt):
    """
    RICKER generate a ricker wavelet
    input (dt,period)
    """
    nt = int(2 * pt / dt)
    c = np.zeros(nt)
    t0 = pt / dt
    a_ricker = 4 / pt

    for it in range(0, nt):
        t = ((it + 1) - t0) * dt
        c[it] = -2 * a_ricker * t * math.exp(-((a_ricker * t) ** 2))
    return c
    
def load_datasets(train_size=26000, val_size=1000, test_size=1000, batch_size=64):
    x = torch.cat([torch.load("dataset/x_train.pt"), torch.load("dataset/x_train_2.pt")])
    y = torch.cat([torch.load("dataset/y_train.pt"), torch.load("dataset/y_train_2.pt")])
    c = torch.cat([torch.load("dataset/c_train.pt"), torch.load("dataset/c_train_2.pt")])
    
    x_test = torch.load("dataset/x_test.pt")
    y_test = torch.load("dataset/y_test.pt")
    c_test = torch.load("dataset/c_test.pt")
    
    x_min, x_max, c_min, c_max, y_min, y_max = x[:train_size].min(), x[:train_size].max(), c[:train_size].min(), c[:train_size].max(), y[:train_size].min(), y[:train_size].max()
    
    x = (x-x_min)/(x_max-x_min)
    c = (c-c_min)/(c_max-c_min)
    y = (y-y_min)/(y_max-y_min)
    
    x_test = (x_test-x_min)/(x_max-x_min)
    c_test = (c_test-c_min)/(c_max-c_min)
    y_test = (y_test-y_min)/(y_max-y_min)
    
    train = DataLoader(TensorDataset(x[:train_size], c[:train_size], y[:train_size]), batch_size=batch_size, shuffle=True)
    val = DataLoader(TensorDataset(x[train_size:train_size+val_size], c[train_size:train_size+val_size], y[train_size:train_size+val_size]), batch_size=batch_size, shuffle=False)
    test = DataLoader(TensorDataset(x_test[:test_size], c_test[:test_size], y_test[:test_size]), batch_size=batch_size, shuffle=False)
    return train, val, test, x_min, x_max, c_min, c_max, y_min, y_max

def rnmse(pred, target):
    if isinstance(pred, np.ndarray):
        return (np.mean((pred - target)**2)/np.var(target))**0.5
    else:
        return (torch.mean((pred - target)**2)/torch.var(target))**0.5