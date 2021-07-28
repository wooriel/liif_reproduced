import torch
import torch.nn as nn
import torch.nn.functional as F


class LIIF(nn.Module):
    def __init__(self, encoder):
        super(LIIF, self).__init__()

        self.encoder = encoder
        self.feature = None

    # def feature_unfolding(coordinate):
    #     # convert hr_coord (latent vec) into 2D spectrum
    #     self.

    def forward(self, lr_img, h_coord, cell):
        self.feature = self.encoder(lr_img)
        