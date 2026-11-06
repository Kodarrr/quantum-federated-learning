import torch
import torch.nn as nn
import copy

class Client:
    def __init__(self,x,y):
        self.x=x
        self.y=y

    def train(self,global_model,epochs=2,lr=0.01,batch_size=16):
        model = copy.deepcopy(global_model)

        optimizer = torch.optim.SGD(model.parameters(),lr=lr)
        loss_fn= nn.CrossEntropyLoss()

        for epoch in range(epochs):
            perm = torch.randperm(len(self.x))

            for i in range(0,len(self.x),batch_size):
                idx= perm[i:i+batch_size]
                x_batch = self.x[idx]
                y_batch = self.y[idx]

                optimizer.zero_grad()
                output= model(x_batch)
                loss= loss_fn(output,y_batch)
                loss.backward()
                optimizer.step()

        return model
