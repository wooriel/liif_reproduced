import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, liif, use_le, dims=[256, 256, 256, 256]): # channel, areas, use_le
        super().__init__()

        self.liif = liif
        self.dims = dims
        self.use_le = use_le
        # if self.use_le:
        #     self.mult = 4
        # else:
        #     self.mult = 1
        self.x_size = None

        self.prev_c = 580
        self.prev_c2 = 580
        self.prev_c3 = 580
        self.prev_c4 = 580
                
        self.layers = nn.ModuleList()
        for i in range(len(self.dims)):
            self.layers.append(nn.Linear(in_features=self.prev_c, out_features=self.dims[i]))
            # self.layers.append(nn.Linear(in_features=self.prev_c * self.mult, out_features=self.dims[i] * self.mult))
            self.prev_c = self.dims[i]
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(in_features=self.prev_c, out_features=3))
        # self.layers.append(nn.Linear(in_features=self.prev_c * self.mult, out_features=3 * self.mult))
        # self.layers.append(nn.ReLU())

        self.layers2 = nn.ModuleList()
        for i in range(len(self.dims)):
            self.layers2.append(nn.Linear(in_features=self.prev_c2, out_features=self.dims[i]))
            # self.layers2.append(nn.Linear(in_features=self.prev_c * self.mult, out_features=self.dims[i] * self.mult))
            self.prev_c2 = self.dims[i]
            self.layers2.append(nn.ReLU())
        self.layers2.append(nn.Linear(in_features=self.prev_c2, out_features=3))
        # self.layers2.append(nn.Linear(in_features=self.prev_c * self.mult, out_features=3 * self.mult))

        self.layers3 = nn.ModuleList()
        for i in range(len(self.dims)):
            self.layers3.append(nn.Linear(in_features=self.prev_c3, out_features=self.dims[i]))
            # self.layers3.append(nn.Linear(in_features=self.prev_c * self.mult, out_features=self.dims[i] * self.mult))
            self.prev_c3 = self.dims[i]
            self.layers3.append(nn.ReLU())
        self.layers3.append(nn.Linear(in_features=self.prev_c3, out_features=3))
        # self.layers3.append(nn.Linear(in_features=self.prev_c * self.mult, out_features=3 * self.mult))

        self.layers4 = nn.ModuleList()
        for i in range(len(self.dims)):
            self.layers4.append(nn.Linear(in_features=self.prev_c4, out_features=self.dims[i]))
            # self.layers4.append(nn.Linear(in_features=self.prev_c * self.mult, out_features=self.dims[i] * self.mult))
            self.prev_c4 = self.dims[i]
            self.layers4.append(nn.ReLU())
        self.layers4.append(nn.Linear(in_features=self.prev_c4, out_features=3))
        # self.layers4.append(nn.Linear(in_features=self.prev_c * self.mult, out_features=3 * self.mult))

    def forward(self, low_res, h_coord, cell, device):
        # x = (B, H*W, 580*4)
        # self.low_res = low_res
        # self.h_coord = h_coord
        # self.cell = cell

        x, self.areas, self.prev_c = self.liif(low_res, h_coord, cell, device)

        x_size = x.size() # to turn back to original shape
        # (B, img_H*img_W(2304), 4, C(580))
        # to convert into b*c*h, w <- intended for change shape of batch coordinates
        nan_occured = torch.all(x.isnan())

        # for i, layer in enumerate(self.layers):
        #     x = layer(x)
        # x = x.view(*x_size[:-1], -1)

        x00 = x[:, :, 0, :].view(-1, x_size[-1])
        x00_sz = x00.size()
        x01 = x[:, :, 1, :].view(-1, x_size[-1])
        x10 = x[:, :, 2, :].view(-1, x_size[-1])
        x11 = x[:, :, 3, :].view(-1, x_size[-1])
        # x = x.view(-1, x_size[-1])

        for i0, layer in enumerate(self.layers):
            x00 = layer(x00)
        x00 = x00.view(*x_size[:-2], -1)

        for i1, layer2 in enumerate(self.layers2):
            x01 = layer2(x01)
        x01 = x01.view(*x_size[:-2], -1)

        for i2, layer3 in enumerate(self.layers3):
            x10 = layer3(x10)
        x10 = x10.view(*x_size[:-2], -1)

        for i3, layer4 in enumerate(self.layers4):
            x11 = layer4(x11)
        x11 = x11.view(*x_size[:-2], -1)


        # input: (B * C (2320) 48*48 * C:3(*4), 580)
        # output: (B, C (2320), 4, 3) ?

        if self.use_le:
            # change the order of area: diagonal
            # areas: (B, 2304, 4)
            # temp = self.areas[:, :, 0]
            # self.areas[:, :, 0] = self.areas[:, :, 3]
            # check = self.areas[:, :, 0]
            # check1 = self.areas[:, :, 3]
            # self.areas[:, :, 3] = temp
            # temp = self.areas[:, :, 1]
            # self.areas[:, :, 1] = self.areas[:, :, 2]
            # check2 = self.areas[:, :, 1]
            # self.areas[:, :, 2] = temp
            # check3 = self.areas[:, :, 2]

            area_sz = self.areas.size()
            self.areas.add(1e-9)
            area_sum = self.areas.sum(dim=0).sum(dim=1)
            area_sum_sz = area_sum.size() # 2304
            x_sz = x.size()
            x = x.permute(0, 2, 1, 3)
            self.areas = self.areas.permute(0, 2, 1).div(area_sum).unsqueeze(-1)

            # image = x[:, 0, :, :].mul(self.areas[:, 3, :, :])
            # image = image.add(x[:, 1, :, :].mul(self.areas[:, 2, :, :]))
            # image = image.add(x[:, 2, :, :].mul(self.areas[:, 1, :, :]))
            # image = image.add(x[:, 3, :, :].mul(self.areas[:, 0, :, :]))

            image = x00.mul(self.areas[:, 3, :, :])
            nan_image = torch.all(x.isnan()) # x[:, 0, :, :]
            image = image.add(x01.mul(self.areas[:, 2, :, :])) # x[:, 1, :, :]
            image = image.add(x10.mul(self.areas[:, 1, :, :])) # x[:, 2, :, :]
            image = image.add(x11.mul(self.areas[:, 0, :, :])) # x[:, 3, :, :]

            # image = x.permute(0, 2, 1, 3) * (self.areas.permute(0, 2, 1).div(area_sum)).unsqueeze(-1)
            # x (B, 4, C, 3) / self.areas.div().un (B, 4, C, 1)
            # image = image.sum(dim=1)
            image_sz = image.size()
        else:
            image = x

        return image
