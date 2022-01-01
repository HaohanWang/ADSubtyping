import torch
import torch.utils.data
import torch.utils.data.sampler
from torch.utils.data import DataLoader
import torch.nn.functional as F

import ipyvolume as ipv
import numpy as np
import pandas as pd
from easydict import EasyDict as edict

from os import makedirs
from os.path import basename,join
from glob import glob
from tqdm import tqdm
import math

import dataManagement.dataset  as ds
import mainScripts_torch.model as torch_model
import utility.config_utils as config_utils

import torch.nn as nn


class MinMaxNormalization(object):
    """Normalizes a tensor between 0 and 1"""
    def __call__(self, image): 
        return (image - image.min()) / (image.max() - image.min())

def model_checkpoint(checkpoint_path, feature=None, config=None):
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
    if config is None:
        config = edict()
        config.model = edict()
        config.model.name = 'Conv5_FC3'
        config.model.params = None
    print("Loaded config = ", config)
    model = torch_model.get_model(config)
    model.load_state_dict(checkpoint_dict)
    
    if feature:
        new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        model.classifier = new_classifier
    
    return model

def saliency_map(model, dataloader, batch_size, save_dir, split):

    label_list = []
    sub_list = []
    sess_list = []
    prob_list = []

    model.eval()
    if True:
        
        total_size = len(dataloader.dataset)
        total_step = math.ceil(total_size / batch_size)
        tbar = tqdm(enumerate(dataloader), total=total_step)

        for i, data in tbar:
            
            images = data['image'].cuda()
            labels = data['label'].cuda()
            
            subs = data['participant_id']
            sess = data['session_id']
            label_list.extend(labels.cpu().numpy())
            sub_list.extend(subs)
            sess_list.extend(sess)

            images.requires_grad_()
            outputs = model(images)
            prob = F.softmax(outputs, dim=1)
            prob_list.append(prob.detach().cpu().numpy()[0])

            outputs_max, _ = torch.max(outputs, dim=1)
            outputs_max.backward()
            saliency, _ = torch.max(images.grad.data.abs(), dim=1)
            for j, sal in enumerate(saliency):
                makedirs(join(save_dir, split, subs[j]), exist_ok=True)
                np.save(join(save_dir, split, subs[j], sess[j]+'.npy'), sal.cpu().numpy())

        df = pd.DataFrame(columns=['participant_id', 'session_id', 'diagnosis', 'prob_AD'])
        
        df['participant_id'] = sub_list
        df['session_id'] = sess_list
        df['diagnosis'] = label_list
        df['prob_AD'] = np.stack(prob_list, axis=0)[:,1]

        df.to_csv(join(save_dir,split+'_sailency_info.csv'), index=None)


def get_dataloader(split, batch_size, is_train=False, num_worker=4, transform=MinMaxNormalization(), idx_fold=0):
    dataset = ds.MRIDataset('/home/ec2-user/alzstudy/AlzheimerData/ADNI_CAPS',split = split, transform=transform, idx_fold=idx_fold)
    dataloader = DataLoader(dataset,
                            shuffle=is_train,
                            batch_size=batch_size,
                            drop_last=is_train,
                            num_workers=num_worker,
                            pin_memory=False)
    return dataloader

if __name__ == "__main__":
    
    batch_size = 1
    # save_dir = "saliency_map_dropblock"
    save_dir = "saliency_map_torch"
    makedirs(save_dir, exist_ok=True)

    # model = model_checkpoint("results/policy_eps5e-3_lr1e-5_weight_dropblock/checkpoint/epoch_0010.pth")
    config = config_utils.load("/home/ec2-user/alzstudy/code/AlzheimerDiseaseUnderstanding/mainScripts_torch/policies/policy_2.yml")
    model = model_checkpoint("/home/ec2-user/alzstudy/checkpoints/policy2_1e-5_dr_0.5_eps_5e-3_seed_0/checkpoint/epoch_0010.pth", None, config)
    model.cuda()
    dataloaders = {split:get_dataloader(split, batch_size=batch_size)
                   for split in ['train', 'test', 'val']}
    saliency_map(model, dataloaders['train'], batch_size, save_dir, 'train')
    saliency_map(model, dataloaders['test'], batch_size, save_dir, 'test')
    saliency_map(model, dataloaders['val'], batch_size, save_dir, 'val')
    



