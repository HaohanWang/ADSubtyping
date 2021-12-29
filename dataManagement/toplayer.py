import torch
import torch.utils.data
import torch.utils.data.sampler
from torch.utils.data import DataLoader
import torch.nn.functional as F

import torchattacks
import ipyvolume as ipv
import numpy as np
import pandas as pd
from easydict import EasyDict as edict

import os
from os import makedirs
from os.path import basename,join
from glob import glob
from tqdm import tqdm
import math

from dataset import MRIDataset
from model import get_model

class MinMaxNormalization(object):
    """Normalizes a tensor between 0 and 1"""
    def __call__(self, image): 
        return (image - image.min()) / (image.max() - image.min())

def model_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    checkpoint_dict = {}
    for k, v in checkpoint['state_dict'].items():
        if 'num_batches_tracked' in k:
            continue
        if k.startswith('module.'):
            if True:
                checkpoint_dict[k[7:]] = v
            else:
                checkpoint_dict['feature_extractor.' + k[7:]] = v
        else:
            if True:
                checkpoint_dict[k] = v
            else:
                checkpoint_dict['feature_extractor.' + k] = v
    config = edict()
    config.model = edict()
    config.model.name = 'Conv5_FC3'
    config.model.params = None
    model = get_model(config)
    model.load_state_dict(checkpoint_dict)
    
    model.classifier = torch.nn.Sequential(*list(model.classifier_ori.children())[:-1])
    
    return model


def last_fc(model, dataloader, batch_size, save_dir, split):

    label_list = []
    sub_list = []
    sess_list = []
    prob_list = []

    model.eval()

    total_size = len(dataloader.dataset)
    total_step = math.ceil(total_size / batch_size)
    tbar = tqdm(enumerate(dataloader), total=total_step)

    with torch.no_grad():
        for i, data in tbar:
            
            images = data['image'].cuda()
            labels = data['label'].cuda()
            
            subs = data['participant_id']
            sess = data['session_id']

            outputs = model(images)
            for j, o in enumerate(outputs):
                makedirs(join(save_dir, split, subs[j]), exist_ok=True)
                if not os.path.exists(join(save_dir, split, subs[j], sess[j]+'.npy')):
                    np.save(join(save_dir, split, subs[j], sess[j]+'.npy'), o.cpu().numpy())


def get_dataloader(split, batch_size, is_train=False, num_worker=4, transform=MinMaxNormalization(), idx_fold=0):
    dataset = MRIDataset('ADNI_CAPS',split = split, transform=transform, idx_fold=idx_fold)
    dataloader = DataLoader(dataset,
                            shuffle=is_train,
                            batch_size=batch_size,
                            drop_last=is_train,
                            num_workers=num_worker,
                            pin_memory=False)
    return dataloader

if __name__ == "__main__":
    
    batch_size = 1
    save_dir = "last_fc_output"
    makedirs(save_dir, exist_ok=True)

    model = model_checkpoint("new_results/pgd_eps_5e-3_iter_10_lr_1e-5/checkpoint/epoch_0008.pth")
    model.cuda()
    dataloaders = {split:get_dataloader(split, batch_size=batch_size)
                   for split in ['train', 'test', 'val']}
    last_fc(model, dataloaders['train'], batch_size, save_dir, 'train')
    last_fc(model, dataloaders['test'], batch_size, save_dir, 'test')
    last_fc(model, dataloaders['val'], batch_size, save_dir, 'val')