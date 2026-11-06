from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np

def load_data():
    x,y= load_digits(return_X_y=True)

    x_train, x_test, y_train, y_test = train_test_split(
        x,y,
        test_size=0.25,
        random_state=0
    )

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train= scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    x_train = torch.tensor(x_train, dtype = torch.float32)
    x_test = torch.tensor(x_test, dtype= torch.float32)

    y_train = torch.tensor(y_train)
    y_test= torch.tensor(y_test)

    return x_train, y_train, x_test, y_test

def split_with_missing_digits(y, missing_digits, seed=0):
    
    n_clients = len(missing_digits)
    rng = np.random.default_rng(seed)

    # Dhap 1: shob index shuffle kore n_clients ta shoman bhag
    all_idx = rng.permutation(len(y))
    parts = np.array_split(all_idx, n_clients)

    # Dhap 2: protita bhag theke missing digit gulo bad
    client_indices = []
    for k in range(n_clients):
        idx = parts[k]
        keep = ~np.isin(y[idx], missing_digits[k])   # missing digit na hole True
        client_indices.append(idx[keep])

    return client_indices