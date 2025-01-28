import torch
from torch.nn import Module

class Relu(Module):
    def __init__(self):
        super(Relu, self).__init__()

    def forward(self, input):
        
        quat_components = torch.chunk(input, 4, dim=1)
       

        r, i, j, k = quat_components[0], quat_components[1], quat_components[2], quat_components[3]

        reluR = torch.relu(r) 
        reluI = torch.relu(i) 
        reluJ = torch.relu(j)
        reluK = torch.relu(k) 


        new_input = torch.cat((reluR, reluI, reluJ, reluK), dim=1)

        return new_input
