from qfl.data import load_data
from qfl.models import make_model

x_train, y_train, x_test, y_test = load_data()
model = make_model()
output= model(x_train[:5])
print(output.shape)