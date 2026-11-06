import torch.nn as nn
import pennylane as qml
def make_model():
    model = nn.Sequential(
        nn.Linear(64,32),
        nn.ReLU(),
        nn.Linear(32,10)
    )
    return model

def make_quantum_model(n_qubits=8, n_layers=2, n_classes=10):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev,interface="torch",diff_method="backprop")
    def circuit(inputs,weights):
        qml.AngleEmbedding(inputs,wires=range(n_qubits))
        #training
        qml.StronglyEntanglingLayers(weights,wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(w))for w in range(n_qubits)]

    qlayer = qml.qnn.TorchLayer(circuit, {"weights": (n_layers, n_qubits, 3)})
    model = nn.Sequential(
        qlayer,
        nn.Linear(n_qubits, n_classes)
    )

    return model

def make_small_classical(n_features=8, hidden=7, n_classes=10):
    model = nn.Sequential(
        nn.Linear(n_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, n_classes)
    )
    return model