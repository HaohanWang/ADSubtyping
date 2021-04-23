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

def model_checkpoint(checkpoint_path, feature=None):
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
    
    if feature:
        new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        model.classifier = new_classifier
    
    return model

def single_attack_image(model, attack, dataloader, batch_size, save_dir, split):

    label_list = []
    sub_list = []
    sess_list = []
    prob_list = []
    prob_attack_list = []

    model.train()
    # with torch.no_grad():
    if True:
        
        total_size = len(dataloader.dataset)
        total_step = math.ceil(total_size / batch_size)
        tbar = tqdm(enumerate(dataloader), total=total_step)

        for i, data in tbar:
            
            images = data['image'].cuda()
            labels = data['label'].cuda()
            images_attack = attack(images, labels).cuda()
            
            subs = data['participant_id']
            sess = data['session_id']
            label_list.extend(labels.cpu().numpy())
            sub_list.extend(subs)
            sess_list.extend(sess)

            with torch.no_grad():
                outputs = model(images)
                outputs_attack = model(images_attack)
                prob = F.softmax(outputs, dim=1)
                prob_attack = F.softmax(outputs_attack, dim=1)
                prob_list.append(prob.detach().cpu().numpy())
                prob_attack_list.append(prob_attack.detach().cpu().numpy())

            images_diff_abs = torch.abs(images_attack.squeeze(1)-images.squeeze(1)).detach().cpu().numpy()
            for j, diff in enumerate(images_diff_abs):
                makedirs(join(save_dir, split, subs[j]), exist_ok=True)
                np.save(join(save_dir, split, subs[j], sess[j]+'.npy'), diff)
#                ipv.pylab.clear()
#                ipv.volshow(diff, level=[1,1,1])
#                ipv.pylab.save(join(save_dir, split, subs[j], sess[j]+'.html'))
        
#            if i == 0:
#                break
        prob_list = np.vstack(prob_list)
        prob_attack_list = np.vstack(prob_attack_list)
        df = pd.DataFrame(columns=['participant_id', 'session_id', 'diagnosis', 'prob_AD', 'prob_AD_attack'])
        
        df['participant_id'] = sub_list
        df['session_id'] = sess_list
        df['diagnosis'] = label_list
        df['prob_AD'] = prob_list[:,1]
        df['prob_AD_attack'] = prob_attack_list[:,1]

        df.to_csv(join(save_dir,split+'_attack_info.csv'), index=None)


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
    
    batch_size = 4
    save_dir = "attack_visualization"
    makedirs(save_dir, exist_ok=True)

    model = model_checkpoint("results/policy_eps5e-3_lr1e-5_weight/checkpoint/epoch_0017.pth")
    model.cuda()
    attack = torchattacks.PGD(model, eps = 0.05, alpha = 0.00125, iters=10, random_start=True)
    dataloaders = {split:get_dataloader(split, batch_size=batch_size)
                   for split in ['train', 'test']}
    single_attack_image(model, attack, dataloaders['train'], batch_size, save_dir, 'train')
    single_attack_image(model, attack, dataloaders['test'], batch_size, save_dir, 'test')
    



