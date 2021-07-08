from PIL import Image
import os
from enum import Enum
import downsample as D

import torch
import torch.utils.data.DataSet as DataSet
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
resize = 48
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
trans_train = transforms.Compose(
    [# transforms.ToTensor(),
     transforms.RandomCrop(resize),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     transforms.RandomHorizontalFlip(),
     transforms.RandomVerticalFlip(),
    #  transforms.RandomRotation([-90, 90]),
     ])

trans_val = transforms.Compose(
    [# transforms.ToTensor(), <- 가급적이면 나중에
     transforms.RandomCrop(resize),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

# only has data normalization
trans_test = transforms.Compose(
    [# transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

#--------------------


# first few data: val = 10
def load_div2k(repeat, scale=None):
    data_dir = ''
    sub_name = ''
    start = 800, end=900 # default: val/test
    
    if scale == None:
        if repeat == Dt.TR.value:
            # load HR train
            data_dir = os.path.join(root, tr_hr)
            start, end = 0, 800
        else:
            # load HR valid
            data_dir = os.path.join(root, val_hr)
            if repeat == Dt.VAL.value:
                end = start + first_k
    else:
        assert scale in (2, 3, 4), "The value of \'scale\' should be one of 2, 3, 4"
        sub_name = ''.join(['x', str(scale)])
        if repeat == Dt.TR.value:
            start, end = 0, 800
            if self.scale == 2:
                data_dir = os.path.join(root, tr_x2)
            elif self.scale == 3:
                data_dir = os.path.join(root, tr_x3)
            else: # self.scale == 4
                data_dir = os.path.join(root, tr_x4)
        else:
            if self.scale == 2:
                data_dir = os.path.join(root, val_x2)
            elif self.scale == 3:
                data_dir = os.path.join(root, val_x3)
            else: # self.scale == 4:
                data_dir = os.path.join(root, val_x4)
            if repeat == Dt.VAL.value:
                end = start + first_k

    images = []
    for i in range(start+1, end+1, 1):
        strnum = "%4d".format(i)
        img_name = ''.join([strnum, sub_name, '.png'])
        print(img_name)
        fname = os.path.join(data_dir, sub_name.upper(), img_name)
        print(fname)
        images.append(transforms.ToTensor(Image.open(fname).convert('RGB')))

    return images



class DIV2K(Dataset):
    def __init__(self, train_val_test, scale=None): #, transform
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
        self.images = load_div2k(self.repeat)
        if scale != None: # use pair of images to downsample
            self.lr_img = load_div2k(self.repeat, scale) # image array
        # self.transform = transform

    def __getitem__(self, idx):
        hr_img = self.images[idx % len(self.images)]
        if self.scale == None:
            # call downsampling function
        else:
            lr_img = self.lr_img[idx % len(self.lr_img)]
            # call downsampling function
            

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