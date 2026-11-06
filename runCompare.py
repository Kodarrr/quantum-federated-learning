from qfl.data import load_quantum_data, split_with_missing_digits, include_digits
from qfl.models import make_quantum_model, make_small_classical
from qfl.client import Client
from qfl.server import fedavg, accuracy


MODEL = "quantum"          # "quantum" ba "classical"
ROUNDS = 50

# ---------- data: duita model e same 8 PCA feature ----------
X_train, y_train, X_test, y_test = load_quantum_data(n_qubits=8)

missing_digits = [
    [1, 3, 7],
    [2, 5, 8],
    [4, 6, 9],
]
parts = split_with_missing_digits(y_train.numpy(), missing_digits)
clients = [Client(X_train[idx], y_train[idx]) for idx in parts]

test_137 = include_digits(X_test, y_test, [1, 3, 7])
test_258 = include_digits(X_test, y_test, [2, 5, 8])
test_469 = include_digits(X_test, y_test, [4, 6, 9])

# ---------- model ----------
if MODEL == "quantum":
    global_model = make_quantum_model(n_qubits=8, n_layers=2, n_classes=10)
else:
    global_model = make_small_classical(n_features=8, hidden=7, n_classes=10)

print("model:", MODEL, "| params:", sum(p.numel() for p in global_model.parameters()))

# ---------- FL rounds (ager motoi) ----------
for rnd in range(1, ROUNDS + 1):

    trained_models = []
    data_sizes = []

    for client in clients:
        model = client.train(global_model, epochs=2, lr=0.1)
        trained_models.append(model)
        data_sizes.append(len(client.y))

    new_weights = fedavg(trained_models, data_sizes)
    global_model.load_state_dict(new_weights)

    acc_all = accuracy(global_model, X_test, y_test)
    acc_137 = accuracy(global_model, test_137[0], test_137[1])
    acc_258 = accuracy(global_model, test_258[0], test_258[1])
    acc_469 = accuracy(global_model, test_469[0], test_469[1])

    print(f"round {rnd} | all {acc_all:.3f} | 137 {acc_137:.3f}"
          f" | 258 {acc_258:.3f} | 469 {acc_469:.3f}")