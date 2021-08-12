import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
# used only for exstracting PIL sample images
import math
import torchvision.transforms as transforms
from PIL import Image
from tensorboardX import SummaryWriter
import random

import os
import argparse
import yaml
import time
from tqdm import tqdm
import glob

from dataset import dataset_loader as D
from torch.utils.data import DataLoader
from model import model_loader as M

batch = 1


# copy and paste from train.py
def test_range(sub=0.5, div=0.5):
    # makes the range into 0~1 into -1~1
    # make into tensor so that it can connect gpu
    d_sub = torch.tensor([sub])
    d_div = torch.tensor([div])
    if type == 0: # image
        d_sub.view(1, 1, 1, 1)
        d_div.view(1, 1, 1, 1)
    else: # gt
        d_sub.view(1, 1, 1)
        d_div.view(1, 1, 1)
    
    return d_sub, d_div


# brought from the original code
# def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
#     diff = (sr - hr) / rgb_range
#     if dataset is not None:
#         if dataset == 'benchmark':
#             shave = scale
#             if diff.size(1) > 1:
#                 gray_coeffs = [65.738, 129.057, 25.064]
#                 convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
#                 diff = diff.mul(convert).sum(dim=1)
#         elif dataset == 'div2k':
#             shave = scale # + 6 
#         else:
#             raise NotImplementedError
#         valid = diff[..., shave:-shave, shave:-shave]
#     else:
#         valid = diff
#     mse = valid.pow(2).mean()
#     return -10 * torch.log10(mse)

def calc_psnr(sample, gt, dataset=None, scale=1, rgb_range=1):
    max_color = torch.tensor([256.]).to(device)
    sample_tensor = tensor_to_sample_image(sample, False).mul(max_color)
    gt_tensor = tensor_to_sample_image(gt, False).mul(max_color)
    calculate_mse = nn.MSELoss()
    mseloss = calculate_mse(sample_tensor, gt_tensor)
    psnr = 20. * torch.log10(max_color).item() - 10. * torch.log10(mseloss).item()
    return psnr


def batched_predict(model, inp, coord, cell, m_type, device, bsize=30000):
    # if m_type == 'baseline':
    #     pred = model(low_res, data[1], data[2])
    # if m_type == 'liif':
    #     pred = model(low_res, data[1], data[2], device)

    with torch.no_grad():
        model.encoder_feat(inp)
        # this will take more training loop
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            val_ql = ql
            coord_sz = coord.size()
            qr = min(ql + bsize, n)
            val_qr = qr
            pred = ''
            if m_type == 'baseline':
                pred = model(inp, coord[:, ql: qr, :], cell[:, ql: qr, :])
            else:
                pred = model(inp, coord[:, ql: qr, :], cell[:, ql: qr, :], device)
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
        pred_sz = pred.size()
    return pred


def tensor_to_sample_image(sample, toimg=True):
    # change tensor size of (1, 9216, 3) into PIL image
    # used sadd for sample add

    sample_sz = sample.size() # B, 2304, 3
    sample = sample.squeeze(0)
    # square = int(math.sqrt(sample.size(-2)))
    # sample = sample.transpose(-2, -1).view(sample_sz[0], 3, square, square).squeeze(0)
    # convert to PIL image after detach
    if toimg:
        to_pil = transforms.ToPILImage()
        # sample_sz = sample.size()
        # index = random.randrange(0, 48)
        sample_image = to_pil(sample.detach().cpu())
        return sample_image
    else:
        return sample


def write_log(stuff, log_file_name):
    if log_file_name is not None:
        # 'a' adding
        with open(log_file_name, 'a') as f:
            f.write(stuff)
        f.close()


def test(test_loader, model, m_type, dset_type, device, save_name, bestpoint):
    log_file_path = os.path.join(save_name, 'test_log.txt')

    # load checkpoint
    assert bestpoint is not None
    model.load_state_dict(bestpoint['model'])
    model = model.to(device)

    lsub, ldiv = test_range()
    gsub, gdiv = lsub.clone(), ldiv.clone()
    lsub = lsub.to(device)
    ldiv = ldiv.to(device)
    gsub = gsub.to(device)
    gdiv = gdiv.to(device)

    
    dset_name, dscale = None, 1
    dset_name, dscale = dset_type.split('_')
    dscale = int(dscale)
    assert dset_name in ['div2k', 'benchmark', None]

    total_psnr = 0
    num_img = 0

    with torch.no_grad():
        name = 0
        for i, data in tqdm(enumerate(test_loader)):
            # to extract images
            once = False
            if i % 10 == 0:
                once = True
                name += 1
            data[0] = data[0].to(device)
            data[1] = data[1].to(device)
            data[2] = data[2].to(device)
            data[3] = data[3].to(device)

            low_res = data[0].sub(lsub).div(ldiv)
            

            # pred = batched_predict(model, low_res, data[1], data[2], m_type, device)
            
            if m_type == 'baseline':
                pred = model(low_res, data[1], data[2])
            else:
                pred = batched_predict(model, low_res, data[1], data[2], m_type, device)

            pred = pred.mul(gdiv).add(gsub)
            pred.clamp(0, 1)

            if dset_name is not None:
                lh, lw = low_res.shape[-2:]
                s = math.sqrt(data[1].shape[1] / (lh * lw))
                sr_shape = [low_res.size(0), round(lh * s), round(lw * s), 3]
                pred = pred.view(*sr_shape).permute(0, 3, 1, 2).contiguous()
                pred_sz = pred.size()
                gt = data[3].view(*sr_shape).permute(0, 3, 1, 2).contiguous()
                if once:
                    sample_image = tensor_to_sample_image(pred)
                    file_name = "{}.jpg".format(str(name)) # str(NI-name_index)+"-"+
                    sample_image_path = os.path.join(save_name, file_name)
                    sample_image.save(sample_image_path)

                    gt_image = tensor_to_sample_image(gt)
                    gt_file_name = "gt{}.jpg".format(str(name)) #  str(NI-name_index)+"-"+
                    gt_image_path = os.path.join(save_name, gt_file_name)
                    gt_image.save(gt_image_path)

            # sr, hr, dataset=None, scale=1, rgb_range=1
            psnr = calc_psnr(pred, gt, dataset=dset_name, scale=dscale)
            total_psnr += psnr
            num_img += low_res.size(0)

            to_log = 'val {:.4f}'.format(total_psnr/num_img)
            print(to_log)
            write_log(to_log, log_file_path)

    to_log = 'result: {:.4f}'.format(total_psnr/num_img)
    print(to_log)
    write_log(to_log, log_file_path)

    return total_psnr/num_img



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="instruction for loading dataset and model. Default example:\
        python train.py --config ...")
    # parser.add_argument("--train", type=str, default="train", help="Flag for train or test")
    parser.add_argument("--config", type=str, default="", help="Write a code for specific test")
    parser.add_argument("--model", type=str, help="If there is saved training, wrote the path of config file of that model")
    parser.add_argument("--gpu", type=int, default=0, help="Type the number of GPU that you will use")
    parser.add_argument("--save", type=str, help="Saving place")

    args = parser.parse_args()

    # load test config file
    f = open(args.config, 'r')
    loaded_yaml = yaml.load(f, Loader=yaml.FullLoader)
    print(loaded_yaml)

    test_dataset, _ = D.read_yaml(loaded_yaml)
    # dset = loaded_yaml['datasets']
    # d_type = dset['type']

    # batch_size=1
    test_loader = DataLoader(test_dataset, batch_size=batch, num_workers=8, pin_memory=True) # num_worker=8

    cuda_type = ''.join(['cuda:', str(args.gpu)])
    device = cuda_type if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    # load model
    assert args.save is not None
    save_name = args.save
    best_path = os.path.join(save_name, "epoch-best.pth")
    if os.path.exists(best_path):
        # load bestpoint
        bestpoint = torch.load(best_path)
        # load the device from the name?
        # go to config file in save_name and use that config..
        # I will change this part, test works only if the model is saved as default file name
        folder_name = args.save.split('/')[-1] + ".yaml"
        f_config = os.path.join("./config/train", folder_name)
        # f_config = os.path.join(args.save, "*.yaml")
        # f_saved_config = glob.glob(f_config)
        # assert len(f_saved_config) == 1
        f_model = open(f_config, 'r')
        loaded_model = yaml.load(f_model, Loader=yaml.FullLoader)
        print(loaded_model)
        model_dic = loaded_model['model']
        model = M.read_yaml(loaded_model)
        model_type = model_dic['type']

        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        if n_gpus > 1:
            model = nn.parallel.DataParallel(model)
        
        # get d_type (dataset and scale) from the name
        config_fname = args.config.split('/')[-1].replace('.yaml', '')
        final_psnr_val = test(test_loader, model, model_type, config_fname, device, save_name, bestpoint)