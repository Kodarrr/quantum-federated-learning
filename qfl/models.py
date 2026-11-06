import torch.nn as nn
def make_model():
    model = nn.Sequential(
        nn.Linear(64,32),
        nn.ReLU(),
        nn.Linear(32,10)
    )
    return model