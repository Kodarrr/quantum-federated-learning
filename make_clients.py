import numpy as np
from qfl.data import load_data, split_with_missing_digits
from qfl.client import Client

X_train, y_train, X_test, y_test = load_data()

missing_digits = [
    [1, 6, 8],
    [2, 5, 9],
    [0, 3, 7],
    [4, 6, 9],
    [1, 2, 8],
]

parts = split_with_missing_digits(y_train.numpy(), missing_digits)

clients =[]
for idx in parts:
    clients.append(Client(X_train[idx],y_train[idx]))

for k,c in enumerate(clients):
    counts = np.bincount(c.y.numpy(),minlength=10)
    print(f"client {k}: {len(c.y):4d}pictues | digits 0-9: {counts.tolist()}")