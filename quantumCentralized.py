from qfl.data import load_quantum_data
from qfl.models import make_quantum_model
from qfl.client import Client
from qfl.server import accuracy

X_train, y_train, X_test, y_test = load_quantum_data(n_qubits=8)

model = make_quantum_model(n_qubits=8, n_layers=2)
print("before:", round(accuracy(model, X_test, y_test), 3))
# print(model(X_test[:2]).shape)

one_client = Client(X_train, y_train)
model = one_client.train(model, epochs=20, lr=0.1)

print("after:", round(accuracy(model, X_test, y_test), 3))