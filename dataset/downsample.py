import torch
import torchvision
import torchvision.transforms as transforms

import os
from PIL import Image


def is_flip(p=0.5, length=3):
    # helper function for augment_img
    # flip = list of True/False ... (total length element)
    flip = []
    for i in range(length):
        flip.append(torch.rand(1).item() < p)
    return flip


def rand_left_top(img_size, crop_size):
    ih = img_size[0]
    iw = img_size[1]
    ch = crop_size[0]
    cw = crop_size[1]
    left = torch.randint(0, iw - cw, size=(1, )).item()
    top = torch.randint(0, ih - ch, size=(1, )).item()
    return left, top


def crop_img(img, lt, sz, mult): # img
    # pil img: size of [W, H]
    # lt: top left point in list [left, top]
    # sz: crop size in list [w, h]
    # mult: scale (int)

    # lt
    left = round(lt[0] * mult)
    top = round(lt[1] * mult)
    right = round((lt[0] + sz[0]) * mult) # W + w
    bottom = round((lt[1] + sz[1]) * mult) # H + h
    # print("{}, {}, {}, {}".format(left, top, right, bottom))

    h, w = img.size()[-2:]
#     print("{}, {}".format(w, h))
    assert right < w and bottom < h,\
        "Coordinate and/or scale out of range"
    return img[:, top:bottom, left:right]
    # return img.crop((left, top, right, bottom))


def augment_tensor(t, flip):
    trans = ''
    if flip[0]:
        trans += 'lr '
        t = t.fliplr()
    if flip[1]:
        trans += 'hb '
        t = t.flipud()
    if flip[2]: # Use ROTATE_90 instead?
        trans += 'trans '
        # I fixed transpose(0, 1) to transpose(1, 2)
        t = t.transpose(1, 2)
    return t, trans


def augment_img(img, flip):
    trans = ''
    if flip[0]:
        trans += 'lr '
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if flip[1]:
        trans += 'hb '
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    if flip[2]: # Use ROTATE_90 instead?
        trans += 'trans '
        img = img.transpose(Image.TRANSPOSE)
    return img, trans