import torch
import torch.nn as nn
import torch.nn.init as init

class MLP(torch.nn.Module):
    
    def __init__(self, layer_sizes, type='tanh', last_activation=False):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            # layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < (len(layer_sizes) - 2 if not last_activation else len(layer_sizes) - 1):  # add an activation function when not in the last layer
                if type=='tanh':
                    layers.append(nn.Tanh())
                if type=='relu':
                    layers.append(nn.ReLU())
                if type=='gelu':
                    layers.append(nn.GELU())
                # layers.append(nn.ReLU())
        self.linears = nn.Sequential(*layers)
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # use xavier_uniform_ initialization
                # init.xavier_uniform_(module.weight)
                # or，use xavier_normal_ initialization
                init.xavier_normal_(module.weight)
                # set all biases to 0
                init.zeros_(module.bias)
                
    def forward(self, x):
        return self.linears(x)