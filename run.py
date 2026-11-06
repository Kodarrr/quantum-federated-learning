from qfl.data import load_data, split_with_missing_digits,include_digits
from qfl.models import make_model
from qfl.client import Client
from qfl.server import fedavg, accuracy

X_train, y_train, X_test, y_test = load_data()

missing_digits = [
    [1, 3, 7],   # client 0 pabe na
    [2, 5, 8],   # client 1 pabe na
    [4, 6, 9],   # client 2 pabe na
]

parts = split_with_missing_digits(y_train.numpy(), missing_digits)
# 3 clients in a list
clients = [Client(X_train[idx], y_train[idx]) for idx in parts]

test_137 = include_digits(X_test, y_test, [1, 3, 7])
test_258 = include_digits(X_test, y_test, [2, 5, 8])
test_469 = include_digits(X_test, y_test, [4, 6, 9])

global_model = make_model()

for rnd in range(1,21):
    trained_models = []
    data_sizes=[]

    for client in clients:
        model = client.train(global_model)

        trained_models.append(model)
        data_sizes.append(len(client.y))

    new_weights= fedavg(trained_models,data_sizes)
    global_model.load_state_dict(new_weights)

    acc_all = accuracy(global_model,X_test,y_test)
    acc_137 = accuracy(global_model,test_137[0],test_137[1])
    acc_258 = accuracy(global_model, test_258[0], test_258[1])
    acc_469 = accuracy(global_model, test_469[0], test_469[1])
    print("round", rnd)
    print("  all digits :", round(acc_all, 3))
    print("  digit 1,3,7:", round(acc_137, 3))
    print("  digit 2,5,8:", round(acc_258, 3))
    print("  digit 4,6,9:", round(acc_469, 3))