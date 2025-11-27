import torch
from torch.utils.data import DataLoader, TensorDataset

def load_datasets():
    x = torch.cat([torch.load("dataset/x_train.pt"), torch.load("dataset/x_train_2.pt")])
    y = torch.cat([torch.load("dataset/y_train.pt"), torch.load("dataset/y_train_2.pt")])
    c = torch.cat([torch.load("dataset/c_train.pt"), torch.load("dataset/c_train_2.pt")])
    
    x_test = torch.load("dataset/x_test.pt")
    y_test = torch.load("dataset/y_test.pt")
    c_test = torch.load("dataset/c_test.pt")
    
    x_min, x_max, c_min, c_max, y_min, y_max = x[:20000].min(), x[:20000].max(), c[:20000].min(), c[:20000].max(), y[:20000].min(), y[:20000].max()
    
    x = (x-x_min)/(x_max-x_min)
    c = (c-c_min)/(c_max-c_min)
    y = (y-y_min)/(y_max-y_min)
    
    x_test = (x_test-x_min)/(x_max-x_min)
    c_test = (c_test-c_min)/(c_max-c_min)
    y_test = (y_test-y_min)/(y_max-y_min)
    
    train = DataLoader(TensorDataset(x[:20000], c[:20000], y[:20000]), batch_size=32, shuffle=True)
    val = DataLoader(TensorDataset(x[20000:], c[20000:], y[20000:]), batch_size=32, shuffle=False)
    test = DataLoader(TensorDataset(x_test, c_test, y_test), batch_size=32, shuffle=False)
    return train, val, test, x_min, x_max, c_min, c_max, y_min, y_max