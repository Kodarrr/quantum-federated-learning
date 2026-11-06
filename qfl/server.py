import torch
def fedavg(models,sizes):
    total = sum(sizes)
    states = [m.state_dict() for m in models]
    new_state = {}

    for key in states[0]:
        result =0
        for i in range(len(states)):
            share = sizes[i]/total
            result = result+ states[i][key]*share
        new_state[key]=result
    return new_state
def accuracy(model, X, y):
    with torch.no_grad():
        pred = model(X).argmax(dim=1)
        return (pred == y).float().mean().item()