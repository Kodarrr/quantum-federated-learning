from qfl.data import load_quantum_data
from qfl.models import make_small_classical
from qfl.client import Client
from qfl.server import accuracy

# quantum er SAME data: 8 PCA feature
X_train, y_train, X_test, y_test = load_quantum_data(n_qubits=8)

model = make_small_classical(n_features=8, hidden=7, n_classes=10)
print("params:", sum(p.numel() for p in model.parameters()))
print("before:", round(accuracy(model, X_test, y_test), 3))

one_client = Client(X_train, y_train)
model = one_client.train(model, epochs=20, lr=0.1)

print("after:", round(accuracy(model, X_test, y_test), 3))