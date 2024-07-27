import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from dataset import get_dataset
from unet import UNet 
from models import get_model
from tqdm import tqdm
import fire
import logging
from collections import OrderedDict
import os
import random
import numpy as np
import time as mytime
from torchvision.transforms import v2
from torchvision import datasets, transforms
from train_meta import train_meta


def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(train_path,
        val_path,
        model_name='resnet18',
        bit_w=2, bit_a=2,
        iterations=20000,
        iterations_meta=500,
        num_samples=1024,
        batch_size=32,
        use_wandb=False,
        alpha=1, beta=1, gamma=1, thereshold=0.2,
        lr_T_alpha=5e-6,
        setting=None):
    start_time = mytime.time()
    if use_wandb:
        import wandb
        wandb.init(project='MetaAug', name=f'{model_name}_w{bit_w}_a{bit_a}_{iterations_meta}_alp{alpha}be{beta}gam{gamma}threshold{thereshold}')
    # sample calibration data from train set of ImageNet
    dataset = get_dataset(train_path, num_samples)
    val_set = get_dataset(val_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    data_loader_val = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    # define transformation network
    Transform_model = UNet(n_channels=3, n_classes=3, bilinear=False)
    Transform_model = Transform_model.cuda()
    # load the transformation network
    Transform_model.load_state_dict(torch.load("warmup/unet.pt"))
    # define the quantized model
    available_models = ('resnet18', 'resnet50', 'mobilenetv2', 'regnetx_600m', 'regnetx_3200m', 'mnasnet2.0', 'mnasnet1.0', 'mobilenetb')
    assert model_name in available_models, f'{model_name} not exist!'
    model = get_model(model_name, pretrained=True).cuda().eval()
    for param in Transform_model.parameters():
        param.requires_grad = True


    train_meta(Transform_model, model, data_loader, data_loader_val, val_set, bit_w=bit_w, bit_a=bit_a, iterations=iterations, iterations_meta=iterations_meta, lr_T_alpha=lr_T_alpha, use_wandb=use_wandb, alpha=alpha, beta=beta, gamma=gamma, thereshold=thereshold, setting=setting, num_samples=num_samples)
    print("time for reconstruct", mytime.time()- start_time)
    

if __name__ == '__main__':
    seed_all(1029)
    fire.Fire(main)