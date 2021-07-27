import torch
import torch.nn as nn
import torch.nn.functional as F

# The baseline just samples the rgb value using grid_sample
class Baseline(nn.Module):
    def __init__(self, encoder):
        super().__init__()

        self.encoder = encoder

    def forward(self, inp, coord, cell):
        feature = self.encoder(inp)
        # unflatten coordinate
        feature_size = feature.size()
        # to_return = F.grid_sample(feature, coord.flip(-1).unsqueeze(0),
        #         mode='nearest', align_corners=False)
        # to_return_size = to_return.size()
        # return to_return
        ret = F.grid_sample(feature, coord.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        ret_size = ret.size()
        return ret
