import torch
import torch.nn as nn
import torch.nn.functional as F

D = 32 # 16
C = 256 # 64
K = 5 # 3
# scale of the image = 2, 3, 4
# H = W = 48

# bias removable
class ResBlock(nn.Module):
    def __init__(self, channel, kernel_size, scale, bias=True):
        super(ResBlock).__init__()

        self.c = channel
        self.k = kernel_size
        self.scale = scale
        self.bias = bias

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=self.c, out_channels=self.c, kernel_size=self.k,
                        stride=1, padding=self.k>>1, bias=self.bias),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.c, out_channels=self.c, kernel_size=self.k,
                        stride=1, padding=self.k>>1, bias=self.bias)
        )

    def forward(self, x):
        ident = x
        output = torch.mul(self.block(x), scale)
        output += ident

        return output

# two types of encoder: EDSR-baseline and RDN (no upsampling)
# mean shift is not used
class Encoder(nn.Module):
    def __init__(self, depth, channel, kernel_size,
                   res_scale, img_scale, use_upsampling):
        super(Encoder).__init__()

        self.d = depth
        self.c = channel
        self.k = kernel_size
        self.res_s = res_scale
        self.img_s = img_scale
        self.upsample = use_upsampling

        # First Convolution Layer
        self.conv = nn.Conv2d(in_channels=3, out_channels=self.c,
                              kernel_size=self.k, padding=self.k>>1)

        # ResNet Layers
        self.blocks = nn.ModuleList()
        for i in ramge(self.d):
            self.blocks.append(ResBlock(self.c, self.k, self.res_s))

        # Original Upsampling Layer(s):
        # convolution(make scale^2 # of filter) + shuffle(rearrange in scaled manner)?
        # Upsampling function: simple interpolation

        self.up = nn.ModuleList()
        while img_scale % 2 == 0:
            img_scale = img_scale >> 1
            self.up.append(nn.Conv2d(in_channels=self.c, out_channels=self.c*4,
                                     kernel_size=3, padding=1))
            self.up.append(nn.PixelShuffle(2))
            # self.up.append(nn.Upsample(scale_factor=2, mode='bicubic'))

        # in original code, it is either 2^n or 3: 2, 3, 4 and 8(possibly)
        while img_scale % 3 == 0:
            img_scale = img_scale//3
            self.up.append(nn.Conv2d(in_channels=self.c, out_channels=self.c*9,
                                     kernel_size=3, padding=1))
            self.up.append(nn.PixelShuffle(3))
            # self.up.append(nn.Upsample(scale_factor=3, mode='bicubic'))
        if scale != 1:
            raise Exception("Not available scale")

        # Last Convolution Layer (Upsample)
        # is self.out_c really needed?
        if self.upsample:
            self.convl = nn.Conv2d(in_channels=self.c, out_channels=3,
                                   kernel_size=self.k, padding=self.k>>1)

    def forward(self, x):
        x = self.conv(x)
        ident = x
    
        for i in range(self.d):
            x = self.blocks[i](x)
        x += ident

        if self.upsample:
            for j, name in enumerate(self.up):
                x = self.up[i](x)
            x = self.convl(x)
            
        return x

    # def load_state_dict(self)