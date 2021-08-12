import torch
import torch.nn as nn
import torch.nn.functional as F

# import torch.autograd.profiler as profiler

# borrowed code to check layers
def hook_fn(m, i, o):

    for grad in i:
        if grad is not None and torch.any(torch.isnan(grad)):
            print(f"{m} input----")
            print(grad)
            if isinstance(m, nn.BatchNorm1d):
                print("running_mean:", m.running_mean,"var:", m.running_var, "b:", m.num_batches_tracked)

    for grad in o:
        if grad is not None and torch.any(torch.isnan(grad)):
            for p in m.parameters():
                print(p.dtype)
            print("output---")
            print(grad)
            if isinstance(m, nn.BatchNorm1d):
                print("running_mean:", m.running_mean,"var:", m.running_var, "b:", m.num_batches_tracked)

class MLP(nn.Module):
    def __init__(self, liif, features, ablation, dims=[256, 256, 256, 256]): # channel, areas, use_le
        super().__init__()

        self.liif = liif
        self.dims = dims
        self.use_fu = ablation[0]
        self.use_le = ablation[1]
        self.use_cd = ablation[2]
        # if self.use_le:
        #     self.mult = 4
        # else:
        #     self.mult = 1
        self.x_size = None

        self.prev_c = features # num_features
        if self.use_fu:
            self.prev_c *= 9
        if self.use_le:
            self.prev_c += 2
        if self.use_cd:
            self.prev_c += 2
        self.prev_c2 = self.prev_c
        self.prev_c3 = self.prev_c
        self.prev_c4 = self.prev_c

        self.encod_called = False
                
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
        for j in range(len(self.dims)):
            self.layers2.append(nn.Linear(in_features=self.prev_c2, out_features=self.dims[j]))
            # self.layers2.append(nn.Linear(in_features=self.prev_c * self.mult, out_features=self.dims[i] * self.mult))
            self.prev_c2 = self.dims[j]
            self.layers2.append(nn.ReLU())
        self.layers2.append(nn.Linear(in_features=self.prev_c2, out_features=3))
        # self.layers2.append(nn.Linear(in_features=self.prev_c * self.mult, out_features=3 * self.mult))

        self.layers3 = nn.ModuleList()
        for k in range(len(self.dims)):
            self.layers3.append(nn.Linear(in_features=self.prev_c3, out_features=self.dims[k]))
            # self.layers3.append(nn.Linear(in_features=self.prev_c * self.mult, out_features=self.dims[i] * self.mult))
            self.prev_c3 = self.dims[k]
            self.layers3.append(nn.ReLU())
        self.layers3.append(nn.Linear(in_features=self.prev_c3, out_features=3))
        # self.layers3.append(nn.Linear(in_features=self.prev_c * self.mult, out_features=3 * self.mult))

        self.layers4 = nn.ModuleList()
        for l in range(len(self.dims)):
            self.layers4.append(nn.Linear(in_features=self.prev_c4, out_features=self.dims[l]))
            # self.layers4.append(nn.Linear(in_features=self.prev_c * self.mult, out_features=self.dims[i] * self.mult))
            self.prev_c4 = self.dims[l]
            self.layers4.append(nn.ReLU())
        self.layers4.append(nn.Linear(in_features=self.prev_c4, out_features=3))
        # self.layers4.append(nn.Linear(in_features=self.prev_c * self.mult, out_features=3 * self.mult))

    def encoder_feat(self, low_res):
        self.liif.encod_feat(low_res)

    def forward(self, low_res, h_coord, cell, device):

        # x = (B, H*W, 580*4)
        # self.low_res = low_res
        # self.h_coord = h_coord
        # self.cell = cell
        x = ''
        if self.encod_called == False:
            x, self.areas, self.prev_c = self.liif(low_res, h_coord, cell, device)
        else:
            x, self.areas, self.prev_c = self.liif.help_forward(h_coord, cell, device)

        x_size = x.size() # to turn back to original shape
        # (B, img_H*img_W(2304), 4, C(580))
        # to convert into b*c*h, w <- intended for change shape of batch coordinates
        # nan_occured = torch.all(x.isnan())

        # for i, layer in enumerate(self.layers):
        #     x = layer(x)
        # x = x.view(*x_size[:-1], -1)

        x00 = x[:, :, 0, :].view(-1, x_size[-1])
        x00_sz = x00.size()
        x01 = x[:, :, 1, :].view(-1, x_size[-1])
        x01_sz = x01.size()
        x10 = x[:, :, 2, :].view(-1, x_size[-1])
        x10_sz = x10.size()
        x11 = x[:, :, 3, :].view(-1, x_size[-1])
        x11_sz = x11.size()
        # x = x.view(-1, x_size[-1])

        for i0, layer in enumerate(self.layers):
            input_x00 = x00
            # with profiler.profile(with_stack=True, profile_memory=True) as prof:
            x00 = layer(x00)
            # h = hook_fn(layer, input_x00, x00)
        x00 = x00.view(*x_size[:-2], -1)

        for i1, layer2 in enumerate(self.layers2):
            input_x01 = x01
            x01 = layer2(x01)
            # h = hook_fn(layer2, input_x01, x01)
        x01 = x01.view(*x_size[:-2], -1)

        for i2, layer3 in enumerate(self.layers3):
            input_x10 = x10
            x10 = layer3(x10)
            # h = hook_fn(layer3, input_x10, x10)
        x10 = x10.view(*x_size[:-2], -1)

        for i3, layer4 in enumerate(self.layers4):
            input_x11 = x11
            x11 = layer4(x11)
            # h = hook_fn(layer4, input_x11, x11)
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

            # area_sz = self.areas.size() # B, H*W, 4
            self.areas = self.areas.add(1e-9)
            area_sum = self.areas.sum(dim=2, keepdim=True).unsqueeze(-1)
            # is_four = torch.all(area_sum.eq(torch.tensor([4.0000]).to(device)))
            # area_sum_sz = area_sum.size() # B 2304 1
            # torch.set_printoptions(edgeitems=100)
            # print(area_sum)
            # torch.set_printoptions(edgeitems=3)
            # x_sz = x.size()
            x = x.permute(0, 2, 1, 3) # B, 4, H*W, C
            # area_sz2 = self.areas.size()
            self.areas = self.areas.unsqueeze(-1).div(area_sum).permute(0, 2, 1, 3)
            area_sz3 = self.areas.size()

            # image = x[:, 0, :, :].mul(self.areas[:, 3, :, :])
            # image = image.add(x[:, 1, :, :].mul(self.areas[:, 2, :, :]))
            # image = image.add(x[:, 2, :, :].mul(self.areas[:, 1, :, :]))
            # image = image.add(x[:, 3, :, :].mul(self.areas[:, 0, :, :]))

            # nan_image = torch.all(x.isnan())
            # nan_area = torch.all(self.areas.isnan())
            # ar11 = self.areas[:, 3, :, :].sum(0)
            # ar10 = self.areas[:, 2, :, :].sum(0)
            # ar01 = self.areas[:, 1, :, :].sum(0)
            # ar00 = self.areas[:, 0, :, :].sum(0)
            image = x00.mul(self.areas[:, 3, :, :]) # x[:, 0, :, :]
            image = image.add(x01.mul(self.areas[:, 2, :, :].sum(0))) # x[:, 1, :, :]
            image = image.add(x10.mul(self.areas[:, 1, :, :].sum(0))) # x[:, 2, :, :]
            image = image.add(x11.mul(self.areas[:, 0, :, :].sum(0))) # x[:, 3, :, :]

            # image = x.permute(0, 2, 1, 3) * (self.areas.permute(0, 2, 1).div(area_sum)).unsqueeze(-1)
            # x (B, 4, C, 3) / self.areas.div().un (B, 4, C, 1)
            # image = image.sum(dim=1)
            image_sz = image.size()
        else:
            image = x

        # print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_gpu_time_total', row_limit=5))

        return image
