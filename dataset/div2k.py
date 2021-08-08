from PIL import Image
import os
from enum import Enum
import dataset.downsample as D
import math
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# <Configurations>---

# datatype = repeat_value
class Dt(Enum): # datatype
    TR = 20
    VAL = 160 # validation
    TEST = 1

# Directory: high resolution(HR) / low resolution(LR)
root = os.path.join(os.getcwd(), "data", "div2k")
tr_hr = "DIV2K_train_HR"
val_hr = "DIV2K_valid_HR"
tr_x2 = os.path.join("DIV2K_train_LR_bicubic", "X2")
tr_x3 = os.path.join("DIV2K_train_LR_bicubic-2", "X3")
tr_x4 = os.path.join("DIV2K_train_LR_bicubic-3", "X4")
val_x2 = os.path.join("DIV2K_valid_LR_bicubic", "X2")
val_x3 = os.path.join("DIV2K_valid_LR_bicubic-2", "X3")
val_x4 = os.path.join("DIV2K_valid_LR_bicubic-3", "X4")

# Repeat Dataset
r_train = 20
r_val = 160
r_test = 1

# Part of Dataset Using for validation
first_k = 10

# Downsampling
resize = (48, 48)
sample_q = 2304

#   (implicit-downsampled)
#   train/val: inp_size/sample_q/max (liif, metasr, all ablation)
#   test     : min 6, 12, 18, 24, 30 (out-distribution)
#
#   (implicit-paired)
#   train/val: inp_size/sample_q (x2/x3/x4)
#   test     : 2, 3, 4           (in-distribution)

# Augmentation:
# Original Code: random hor/vert flip / transpose
# rotation sequence

# this transform is applicable to lr of pairs
# random crop of hr would be resize*scale or lr_image*scale
# test - no resize variable, no random crop for lr_image
#--------------------


# first few data: val = 10
def load_div2k(repeat, scale=None):
    data_dir = ''
    sub_name = ''
    start, end = 800, 900 # default: val/test
    
    if scale == None:
        if repeat == Dt.TR.value:
            # load HR train
            data_dir = os.path.join(root, tr_hr)
            start, end = 0, 1 # 800
        else:
            # load HR valid
            data_dir = os.path.join(root, val_hr)
            if repeat == Dt.VAL.value:
                end = start + first_k
    else:
        assert scale in (2, 3, 4), "The value of \'scale\' should be one of 2, 3, 4"
        sub_name = ''.join(['x', str(scale)])
        if repeat == Dt.TR.value:
            start, end = 0, 1 # 800
            if scale == 2:
                data_dir = os.path.join(root, tr_x2)
            elif scale == 3:
                data_dir = os.path.join(root, tr_x3)
            else: # scale == 4
                data_dir = os.path.join(root, tr_x4)
        else:
            if scale == 2:
                data_dir = os.path.join(root, val_x2)
            elif scale == 3:
                data_dir = os.path.join(root, val_x3)
            else: # scale == 4:
                data_dir = os.path.join(root, val_x4)
            if repeat == Dt.VAL.value:
                end = start + first_k

    images = []
    for i in tqdm(range(start+1, end+1, 1)):
        strnum = "{:04d}".format(i)
        img_name = ''.join([strnum, sub_name, '.png'])
        # print(img_name)
        fname = os.path.join(data_dir, sub_name.upper(), img_name)
        # print(fname)
        images.append(transforms.ToTensor()(Image.open(fname).convert('RGB')))

    return images

# old version
# def latent_coord(img_size, ran=[-1, 1], flatten=True):
#     w_x = (ran[1] - ran[0])/ img_size[0]
#     w_y = (ran[1] - ran[0])/ img_size[1]
#     center_x = w_x / 2
#     center_y = w_y / 2
#     coord_x = torch.arange(ran[0]+center_x, ran[1]+center_x, w_x)
#     coord_y = torch.arange(ran[0]+center_y, ran[1]+center_y, w_y)
#     dup_x = coord_x.repeat(img_size[1], 1)
#     dup_y = coord_y.repeat(img_size[0], 1).transpose(0, 1).contiguous()
#     latent = torch.stack([dup_y, dup_x], dim=2) # coord of row first
#     code_latent = latent.transpose(0, 1).contiguous() # coord of column first
#     if flatten:
#         code_latent = code_latent.view(ran[0]*ran[1], 2)
#     return code_latent 

def latent_coord(total_size, ran=[-1, 1], flatten=True):
    img_size = total_size[-2:]
    w_x = (ran[1] - ran[0])/ img_size[1]
    w_y = (ran[1] - ran[0])/ img_size[0]
    center_x = w_x / 2
    center_y = w_y / 2
    coord_x = torch.arange(ran[0]+center_x, ran[1], w_x)
    coord_y = torch.arange(ran[0]+center_y, ran[1], w_y)
    dup_x = coord_x.repeat(img_size[0], 1) # repeat w coord as number of h
    dup_y = coord_y.repeat(img_size[1], 1).transpose(0, 1).contiguous() # repeat h coord as number of w
    latent = torch.stack([dup_x, dup_y], dim=2) # coord of column first (h, w)
    code_latent = latent.transpose(0, 1).contiguous() # coord of row first (h, w)
    if img_size != total_size: # B C H W -> B H W / I just leave as B C H W
        code_latent = code_latent.repeat((*total_size[:-2], 1, 1, 1))
    if flatten:
        code_latent = code_latent.view(*total_size[:-3], ran[0]*ran[1], 2)
        code_latent_size = code_latent.size()
    return code_latent 


class DIV2K(Dataset):
    def __init__(self, train_val_test, scale=None, min_max=None, sample_q=None):
        super(DIV2K, self).__init__()

        # train_val_test is string -> assign repeat value
        # using a repeat value to distinguish between train/val/test
        # repeat: train = 20, val = 160, test = 1
        self.status = train_val_test
        if self.status == 'train':
            self.repeat = Dt.TR.value
        elif self.status == 'validation':
            self.repeat = Dt.VAL.value
        elif self.status == 'test':
            self.repeat = Dt.TEST.value
        else:
            raise Exception("Not a valid option, input should be \
                among \'train\', \'validation\', \'test\'")
        self.scale = scale
        self.min_max = min_max
        self.sample_q = sample_q
        self.images = load_div2k(self.repeat)
        if scale != None: # use pair of images to downsample
            self.lr_img = load_div2k(self.repeat, scale) # image array
        # self.transform = transform

    def __getitem__(self, idx):
        hr_img = self.images[idx % len(self.images)]
        if self.scale == None:
            # call downsampling function (implicit)
            if self.status in ['train', 'validation']:
                scale = torch.rand([1]).mul(self.min_max-1).add(1).item()
                end_limit = (round(resize[0] * scale), round(resize[1] * scale))
                leftTop = D.rand_left_top(hr_img.size()[-2:], end_limit) # low-res
                # size=(height, width) for Resize: resize var should be flipped
                # cut hr first, then resize lr from there
                hr_crop = D.crop_img(hr_img, leftTop, end_limit, 1) # 
                lr_resize = transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC)
                lr_crop = lr_resize(hr_crop)
            else: # test - no resize variable exists
                scale = self.min_max
                height = math.floor(hr_img.shape[-2] / scale + 1e-9) # img -> hr_img
                width = math.floor(hr_img.shape[-1] / scale + 1e-9)
                # size=(height, width) for Resize
                lr_resize = transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BICUBIC)
                lr_crop = lr_resize(hr_img)
                hr_crop = D.crop_img(hr_img, (0, 0), (width, height), scale) # same as itself
            
            # Convert to torch
            # to_tensor = transforms.ToTensor()
            # lr_crop = to_tensor(lr_crop)
            # hr_crop = to_tensor(hr_crop)

            # Data Augmentation
            if self.status == 'train':
                flips = D.is_flip()
                lr_crop, _ = D.augment_tensor(lr_crop, flips)
                hr_crop, _ = D.augment_tensor(hr_crop, flips)
            hr_crop = hr_crop.contiguous() # should be applied to validation as well

            # Sample HR: get latent coordinate and RGB
            hr_coord = latent_coord(hr_crop.size()[-2:])
            hr_crop_sz = hr_crop.size()
            hr_rgb = hr_crop.view(3, -1).transpose(0, 1)

            # sample_q: only for train / ablation, not during test
            # 2304 = 48*48...!
            if self.status in ['train', 'validation']:
                hr_len = len(hr_coord)
                sample_lst = np.random.choice(
                    len(hr_coord), sample_q, replace=False
                )
                hr_coord = hr_coord[sample_lst]
                hr_rgb = hr_rgb[sample_lst]

            # one cell = 4 latent vector <- used in local ensemble
            cell = torch.ones(hr_coord.size()).mul(2)
            size = torch.Tensor([hr_crop.size(0), hr_crop.size(1)])
            cell.div(size)

            return lr_crop, hr_coord, cell, hr_rgb
        
        else:
            lr_img = self.lr_img[idx % len(self.lr_img)]
            # call downsampling function (paired)
            # test / scale does not really change here
            hr_crop, lr_crop = hr_img, lr_img
            if self.status in ['train', 'validation']:
                # Crop Image <- in tensor form
                leftTop = D.rand_left_top(lr_img.size()[-2:], resize)
                lr_crop = D.crop_img(lr_img, leftTop, resize, 1)
                hr_crop = D.crop_img(hr_img, leftTop, resize, self.scale)

            # Convert to torch
            # to_tensor = transforms.ToTensor()
            # lr_crop = to_tensor(lr_crop)
            # hr_crop = to_tensor(hr_crop)

            # Data Augmentation
            if self.status == 'train':
                flips = D.is_flip()
                lr_crop, _ = D.augment_tensor(lr_crop, flips)
                hr_crop, _ = D.augment_tensor(hr_crop, flips)
            hr_crop = hr_crop.contiguous() # should be applied to validation as well

            # Sample HR: get latent coordinate and RGB
            hr_coord = latent_coord(hr_crop.size()[-2:])
            hr_rgb = hr_crop.view(3, -1).transpose(0, 1)
            
            # div2k paired does not do sample
            assert sample_q != None

            # one cell = 4 latent vector <- used in local ensemble
            cell = torch.ones(hr_coord.size()).mul(2)
            size = torch.Tensor([hr_crop.size(0), hr_crop.size(1)])
            cell.div(size)

            return lr_crop, hr_coord, cell, hr_rgb
            

    def __len__(self):
        return len(self.images) * self.repeat
        
        
# transform:
#   training:
#   ✔ augmentation
#   ✔ inp_size: 48
#   ✔ sample_q: 2304
#   ✔ max for liif, metasr / fixed scale for x2/x3/x4
#
#   validation:
#   ✔ inp_size: 48
#   ✔ sample_q: 2304
#   ✔ max for liif, metasr / fixed scale for x2/x3/x4
#
#   test:
#   ✔ min for out-distribution / no number for in-distribution