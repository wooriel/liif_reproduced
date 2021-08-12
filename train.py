import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
# used only for exstracting PIL sample images
import math
import torchvision.transforms as transforms
from PIL import Image
from tensorboardX import SummaryWriter

import os
import argparse
import yaml
import time
from tqdm import tqdm
import random

from dataset import dataset_loader as D
from torch.utils.data import DataLoader
from model import model_loader as M

torch.autograd.set_detect_anomaly(True)


batch = 16 # 16
# epoch_max = 31


def train_range(sub=0.5, div=0.5):
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


def image_range(add=0.5, div=2):
    # make the output range into image 0~1
    d_add = torch.tensor([add])
    d_div = torch.tensor([div])
    if type == 0: # image
        d_add.view(1, 1, 1, 1)
        d_div.view(1, 1, 1, 1)
    else: # gt
        d_add.view(1, 1, 1)
        d_div.view(1, 1, 1)
    
    return d_add, d_div


def tensor_to_sample_image(sample, toimg=True):
    # change tensor size of (1, 9216, 3) into PIL image
    # used sadd for sample add
    radd, rdiv = image_range()
    radd = radd.to(device)
    rdiv = rdiv.to(device)

    # change the pred tensor
    sample = sample.div(rdiv).add(radd)
    # sample = sample.transpose(-2, -1)
    sample_sz = sample.size() # B, 2304, 3
    square = int(math.sqrt(sample.size(-2)))
    sample = sample.transpose(-2, -1).view(batch, 3, square, square).squeeze(0)
    # convert to PIL image after detach
    if toimg:
        to_pil = transforms.ToPILImage()
        sample_sz = sample.size()
        # index = random.randrange(0, 48)
        sample_image = to_pil(sample[random.randrange(0, 16)].detach().cpu())
        return sample_image
    else:
        return sample


def order_tensor_sample(sample):
    to_order = sample.clone()


def calculate_psnr(sample, gt):
    max_color = torch.tensor([256.]).to(device)
    sample_tensor = tensor_to_sample_image(sample, False).mul(max_color)
    gt_tensor = tensor_to_sample_image(gt, False).mul(max_color)
    calculate_mse = nn.MSELoss()
    mseloss = calculate_mse(sample_tensor, gt_tensor)
    psnr = 20. * torch.log10(max_color).item() - 10. * torch.log10(mseloss).item()
    return psnr


def config_to_save_name(string_name):
    string_name = string_name.replace("./config/", "")
    string_name = string_name.replace("train/", "")
    string_name = string_name.replace("test/", "") # needed here?
    string_name = string_name.replace(".yaml", "")
    return string_name


def write_log(stuff, log_file_name):
    if log_file_name is not None:
        # 'a' adding
        with open(log_file_name, 'a') as f:
            f.write(stuff)
        f.close()


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


def get_tensor_info(tensor):
  info = []
  for name in ['requires_grad', 'is_leaf', 'grad_fn', 'grad']:
    info.append(f'{name}({getattr(tensor, name)})')
  info.append(f'tensor({str(tensor)})')
  return ' '.join(info)


def train_val(dt, dv, model, type, device, save_place, cp=None):
    criternion = nn.L1Loss()
    # m_params = list(model.parameters())
    # print (m_params)
    
    # lr need to be decay after 100 epoch by a factor of 0.1
    optimizer = optim.Adam(params=model.parameters(), lr=1e-4)
    # lr_step / epoch setting
    milestone_lst = [200, 400, 600, 800]
    gamma = 0.5
    epoch_max = 1000

    # sample saving name
    freq = 50 # how often save the epoch and image
    amount = epoch_max / freq
    AM = epoch_max / freq
    if type == 'celeb':
        milestone_lst = [100]
        gamma = 0.1
        epoch_max = 200
    scheduler = MultiStepLR(optimizer, milestones=milestone_lst, gamma=gamma)

    # log and tensorboardX
    log_file_path = os.path.join(save_place, 'log.txt')
    writer = SummaryWriter(os.path.join(save_place, 'tensorboard'))
    
    lsub, ldiv = train_range()
    gsub, gdiv = train_range()
    lsub = lsub.to(device)
    ldiv = ldiv.to(device)
    gsub = gsub.to(device)
    gdiv = gdiv.to(device)

    epoch_start = 0
    if cp is not None:
        model.load_state_dict(cp['model'])
        optimizer.load_state_dict(cp['optimizer'])
        epoch_start = cp['epoch']

    # connect to the device
    model = model.to(device)
    # https://github.com/pytorch/pytorch/issues/2830
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    min_val_loss = None
    amount = epoch_max / freq
    AM = epoch_max / freq
    for epoch in range(epoch_start + 1, epoch_max + 1):
        train_start_time = time.time()
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = 0
        min_loss = None
        best_epoch = 0
        save_best_contents = None
        tcount = 0
        curr_rh = 0
        prev_rh = 0
        for i, data in tqdm(enumerate(dt)):
            # data is a list of (lr_crop, hr_coord, cell, gt = hr_rgb)
            data[0] = data[0].to(device)
            data[1] = data[1].to(device)
            data[2] = data[2].to(device)
            data[3] = data[3].to(device)
            

            # input data
            low_res = data[0].sub(lsub).div(ldiv)
            # cell information not needed
            # in case of 'baseline'
            pred = ''
            if type == 'baseline':
                pred = model(low_res, data[1], data[2])
            elif type == 'liif':
                pred = model(low_res, data[1], data[2], device)

            # ground truth data
            gt = data[3].sub(gsub).div(gdiv)

            optimizer.zero_grad()

            loss = criternion(pred, gt)
            train_loss += loss.item()
            writer.add_scalars('loss', {'train': train_loss}, epoch)
            # print('loss', get_tensor_info(loss))
            # loss.retain_grad() 
            # print(loss)
            loss.backward()
            # print('loss_after_backward', get_tensor_info(loss))   
            optimizer.step()

            if min_loss == None or min_loss > loss:
                # update epoch, loss, state_dict()
                save_best_contents = dict(
                    min_loss=loss,
                    best_epoch=epoch,
                    optimizer=optimizer,
                    model=model.state_dict()
                )
            tcount += 1

            if epoch % freq == 0: # epoch % 1 for toy dataset / freq
                save_contents = dict(
                    avg_loss=train_loss/tcount,
                    epoch=epoch,
                    optimizer=optimizer,
                    model=model.state_dict()
                )
                save_num = 'epoch-%d.pth' % epoch
                torch.save(save_contents, os.path.join(save_place, save_num))

        scheduler.step()

        batch_time = time.time()-train_start_time
        to_write = "Epoch [{}/{}]: average train loss={}, count={} time={}\n".format(epoch, epoch_max, train_loss/tcount, tcount, batch_time)
        print(to_write)
        write_log(to_write, log_file_path)

        # for each epoch, log'
        # change this into 10 or 100 later
        save_contents = dict(
            avg_loss=train_loss/tcount,
            epoch=epoch,
            optimizer=optimizer.state_dict(),
            model=model.state_dict()
        )
        # save last epoch
        save_last = 'epoch-last.pth'
        torch.save(save_contents, os.path.join(save_place, save_last))
        # save best epoch here as well
        save_best = 'epoch-best.pth'
        torch.save(save_best_contents, os.path.join(save_place, save_best))

        once = False
        if epoch % 10 == 0:
            once = True
        with torch.no_grad():
            val_start_time = time.time()
            val_loss = 0
            val_psnr = 0
            vcount = 0
            for i, data in tqdm(enumerate(dv)):
                data[0] = data[0].to(device)
                data[1] = data[1].to(device)
                data[2] = data[2].to(device)
                data[3] = data[3].to(device)

              
                ## input data
                low_res = data[0].sub(lsub).div(ldiv)
                # cell information not needed
                pred = ''
                if type == 'baseline':
                    pred = model(low_res, data[1], data[2])
                elif type == 'liif':
                    pred = model(low_res, data[1], data[2], device)

                # ground truth data
                gt = data[3].sub(gsub).div(gdiv)

                # save sample image and gt image
                if amount > 0 and once:
                    once = False
                    amount -= 1
                    # if type == 'liif':
                    #     # something that changes the output into ordered sampled images
                    #     ord_image = order_tensor_sample(pred)
                    sample_image = tensor_to_sample_image(pred)
                    file_name = "{}.jpg".format(str(AM-amount)) # str(NI-name_index)+"-"+
                    sample_image_path = os.path.join(save_name, file_name)
                    sample_image.save(sample_image_path)

                    gt_image = tensor_to_sample_image(gt)
                    gt_file_name = "gt{}.jpg".format(str(AM-amount)) #  str(NI-name_index)+"-"+
                    gt_image_path = os.path.join(save_name, gt_file_name)
                    gt_image.save(gt_image_path)

                
                loss = criternion(pred, gt)
                # print(loss)
                val_loss += loss.item()
                writer.add_scalars('loss', {'train': val_loss}, epoch)
                vcount += 1

                # calculate the psnr value (MSE)
                val_psnr += calculate_psnr(pred, gt)
                writer.add_scalars('psnr', {'val': val_psnr}, epoch)
                # later log the psnr value

            # record the best test loss
            if epoch == epoch_start + 1:
                min_val_loss = val_loss/vcount
            elif val_loss < min_val_loss:
                min_val_loss = val_loss/vcount

            val_time = time.time() - val_start_time
            to_write = "Epoch [{}/{}]: average validation loss={}, avg_psnr={}, count={}, time={}\n".format(epoch, epoch_max, val_loss/vcount, val_psnr/vcount, vcount, val_time)
            print(to_write)

    to_write = "Final best validation loss: {}\n".format(min_val_loss)
    print(to_write)
    write_log(to_write, log_file_path)
    writer.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="instruction for loading dataset and model. Default example:\
        python train.py --config ...")
    parser.add_argument("--train", type=str, default=True, help="Flag for train or test")
    parser.add_argument("--config", type=str, default="", help="Write a code for specific test")
    parser.add_argument("--model", type=str, help="If there is saved training, wrote the path of config file of that model")
    parser.add_argument("--gpu", type=int, default=0, help="Type the number of GPU that you will use")
    parser.add_argument("--save", type=str, help="Saving place")

    args = parser.parse_args()

    f = open(args.config, 'r')
    loaded_yaml = yaml.load(f, Loader=yaml.FullLoader)
    print(loaded_yaml)

    train_dataset, val_dataset = D.read_yaml(loaded_yaml)

    # batch_size=16
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True) # num_worker=2 # 16
    val_loader = DataLoader(val_dataset, batch_size=batch) # 16

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda_type = ''.join(['cuda:', str(args.gpu)])
    device = cuda_type if torch.cuda.is_available() else 'cpu'
    model = M.read_yaml(loaded_yaml)
    # model.to(device)
    model_dic = loaded_yaml['model']
    model_type = model_dic['type']

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    if args.model is not None:
        # load epoch, last_epoch etc
        # load state_dict for the optimizer
        pass
    # actually above objective is done at below
    if args.save is not None:
        save_name = args.save
    else:
        save_name = config_to_save_name(args.config)
        save_name = os.path.join("./save", save_name)
        os.makedirs(save_name, exist_ok=True)
    check_path = os.path.join(save_name, "epoch-last.pth")
    if os.path.exists(check_path):
        # load checkpoint
        checkpoint = torch.load(check_path)
        train_val(train_loader, val_loader, model, model_type, device, save_name, checkpoint)
    else:
        train_val(train_loader, val_loader, model, model_type, device, save_name)


