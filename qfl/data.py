from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
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

def include_digits(X, y, included_digits):
    # shudhu included_digits er chobi gulo rakhe
    keep = torch.isin(y, torch.tensor(included_digits))
    return X[keep], y[keep]

def load_quantum_data(n_qubits=8):
    X, y = load_digits(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    # 64 feature -> n_qubits feature (shudhu train theke fit)
    pca = PCA(n_components=n_qubits)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    # 0 theke pi te scale (rotation angle)
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)

    return X_train, y_train, X_test, y_test
