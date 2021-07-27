import torch
import torch.nn as nn
import torch.optim as optim

import os
import argparse
import yaml
import time
from tqdm import tqdm


from dataset import dataset_loader as D
from torch.utils.data import DataLoader
from model import model_loader as M

from model.encoder import edsr
from model.decoder import baseline

batch = 1 # 16
epoch_start = 0
epoch_max = 10


def train_range(type, sub=0.5, div=0.5):
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


def image_range(type, add=0.5, div=2):
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


def config_to_save_name(string_name):
    string_name = string_name.replace("./config/", "")
    string_name = string_name.replace("train/", "")
    string_name = string_name.replace("test/", "") # needed here?
    string_name = string_name.replace(".yaml", "")
    return string_name
 

def train_val(dt, dv, model, device, save_place, cp=None):
    criternion = nn.L1Loss()
    # m_params = list(model.parameters())
    # print (m_params)
    
    optimizer = optim.Adam(params=model.parameters(), lr=1e-4)
    lsub, ldiv = train_range(dt, 0)
    gsub, gdiv = train_range(dt, 1)
    lsub.to(device)
    ldiv.to(device)
    gsub.to(device)
    gdiv.to(device)

    if cp is not None:
        model.load_state_dict(cp['model'])
        optimizer.load_state_dict(cp['optimizer'])
        epoch_start = cp['epoch']


    min_test_loss = 0
    for epoch in range(epoch_start + 1, epoch_max + 1):
        train_start_time = time.time()

        train_loss = 0
        min_loss = None
        best_epoch = 0
        save_best_contents = None
        tcount = 0
        for i, data in tqdm(enumerate(dt)):
            # data is a list of (lr_crop, hr_coord, cell, gt = hr_rgb)
            # data.to(device)
            # print(data)
            # print(data[0].size())
            # print(data[1].size())
            # print(data[2].size())
            # print(data[3].size())
            data[0].to(device)
            data[1].to(device)
            data[2].to(device)
            data[3].to(device)

            # input data
            low_res = data[0].sub(lsub).div(ldiv)
            pred = model(low_res, data[2], data[3])

            # ground truth data
            gt = data[3].sub(gsub).div(gdiv)

            loss = criternion(pred, gt)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
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

            if epoch % 1 == epoch:
                save_contents = dict(
                    avg_loss=train_loss/tcount,
                    epoch=epoch,
                    optimizer=optimizer,
                    model=model.state_dict()
                )
                save_num = 'epoch-%d.pth' % epoch
                torch.save(save_contents, os.path.join(save_place, save_num))

        batch_time = time.time()-train_start_time
        print("Epoch [{}/{}]: average train loss={}, count={} time={}".format(epoch, epoch_max, train_loss/tcount, tcount, batch_time))
    
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

        with torch.no_grad():
            val_start_time = time.time()
            val_loss = 0
            min_val_loss = None
            for i, data in tqdm(enumerate(dv)):
                data[0].to(device)
                data[1].to(device)
                data[2].to(device)
                data[3].to(device)

                ## input data
                low_res = data[0].sub(lsub).div(ldiv)
                pred = model(low_res, data[2], data[3])

                # ground truth data
                gt = data[3].sub(gsub).div(gdiv)

                loss = criternion(pred, gt)
                val_loss += loss.item()

                if min_val_loss is None:
                    min_val_loss = loss.item()

            val_time = time.time() - val_start_time
            print("Epoch [{}/{}]: average validation loss={}, time={}".format(epoch, epoch_max, val_loss, val_time))

            # record the best test loss
            if epoch == epoch_start:
                min_val_loss = val_loss
            elif val_loss < min_val_loss:
                min_val_loss = val_loss

            # calculate the psnr value



    print("Final best validation loss: {}".format(min_val_loss))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="instruction for loading dataset and model. Default example:\
        python train.py --config ...")
    parser.add_argument("--train", type=str, default="train", help="Flag for train or test")
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
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True) # num_worker=2
    val_loader = DataLoader(val_dataset, batch_size=1)

    cuda_type = ''.join(['cuda', str(args.gpu)])
    device = cuda_type if torch.cuda.is_available() and args.gpu else 'cpu'
    model = M.read_yaml(loaded_yaml)
    model.to(device)

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
        train_val(train_loader, val_loader, model, device, save_name, checkpoint)
    else:
        train_val(train_loader, val_loader, model, device, save_name)