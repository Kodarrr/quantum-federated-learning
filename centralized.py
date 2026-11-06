from qfl.data import load_data
from qfl.models import make_model
from qfl.client import Client
from qfl.server import accuracy

X_train, y_train, X_test, y_test = load_data()

# ekta client, kintu tar kache SHOB data, kono digit missing na
one_client = Client(X_train, y_train)

model = make_model()

# FL e 20 round x 2 epoch = 40 epoch, tai fair tulonar jonno 40 epoch
model = one_client.train(model, epochs=40, lr=0.01)

print("centralized accuracy:", round(accuracy(model, X_test, y_test), 3))