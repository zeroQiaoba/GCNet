import torch
import torch.nn as nn

class FcEncoder(nn.Module):
    def __init__(self, input_dim, layers, dropout=0.5, use_bn=False):
        ''' Fully Connect classifier
            fc+relu+bn+dropout， 最后分类128-4层是直接fc的
            Parameters:
            --------------------------
            input_dim: input feature dim
            layers: [x1, x2, x3] will create 3 layers with x1, x2, x3 hidden nodes respectively.
            dropout: dropout rate
            use_bn: use batchnorm or not
        '''
        super().__init__()
        self.all_layers = []
        for i in range(0, len(layers)):
            self.all_layers.append(nn.Linear(input_dim, layers[i]))
            self.all_layers.append(nn.ReLU())
            if use_bn:
                self.all_layers.append(nn.BatchNorm1d(layers[i]))
            if dropout > 0:
                self.all_layers.append(nn.Dropout(dropout))
            input_dim = layers[i]
        
        self.module = nn.Sequential(*self.all_layers)
    
    def forward(self, x):
        ## make layers to a whole module
        feat = self.module(x)
        return feat