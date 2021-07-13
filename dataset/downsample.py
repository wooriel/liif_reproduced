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
    iw, ih = img_size
    cw = crop_size[0]
    ch = crop_size[1]
    left = torch.randint(0, iw - cw + 1, size=(1, )).item()
    top = torch.randint(0, ih - ch + 1, size=(1, )).item()
    return left, top


def crop_img(img, lt, sz, mult): # img
    # pil img: size of [W, H]
    # lt: top left point in list [left, top]
    # sz: crop size in list [w, h]
    # mult: scale (int)

    # lt
    left = lt[0] * mult
    top = lt[1] * mult
    right = (lt[0] + sz[0]) * mult # W + w
    bottom = (lt[1] + sz[1]) * mult # H + h
    # print("{}, {}, {}, {}".format(left, top, right, bottom))

    w, h = img.size
#     print("{}, {}".format(w, h))
    assert right < w and bottom < h,\
        "Coordinate and/or scale out of range"
    return img.crop((left, top, right, bottom))


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
        t = t.transpose(0, 1)
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