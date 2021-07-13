import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, channel, dims=[256, 256, 256, 256]):
        super.__init__()

        self.dims = dims
        self.prev_c = channel
        
        self.layers = nn.ModuleList()
        for i in range(len(self.dims)):
            self.layers.append(nn.Linear(in_channels=self.prev_c, out_channels=self.dims[i]))
            self.prev_c = self.dims[i]
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(in_channels=self.prev_c, out_channels=3))
        self.layers.append(nn.ReLU())

    def forward(self, x):
        # x has size
        x_size = x.size() # to turn back to original shape
        # to convert into b*c*h, w <- intended for change shape of batch coordinates
        x = x.view(-1, x.shape[-1])
        x = self.layers(x)
        x = x.view(*x_size)

        return x