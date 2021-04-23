import sys
sys.path.append('../')

import torch
import torch.utils.data
import torch.utils.data.sampler
from torch.utils.data import DataLoader
import torch.nn.functional as F

import torchattacks
import numpy as np
import pandas as pd
from easydict import EasyDict as edict

from os import makedirs
from os.path import basename,join
from glob import glob
from tqdm import tqdm
import math

from dataMangement.dataset import MRIDataset
from dataMangement.dataset import get_dataloader
from mainScripts.model import get_model
from utility import config_utils



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
    
    if config == None:
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

def predict(model, dataloader, batch_size, save_dir, split):

    label_list = []
    prob_list = []
    path_list = []
    sub_list = []
    sess_list = []

    model.eval()
        
    total_size = len(dataloader.dataset)
    total_step = math.ceil(total_size / batch_size)
    tbar = tqdm(enumerate(dataloader), total=total_step)

    for i, data in tbar:
        
        images = data['image'].cuda()
        labels = data['label'].cuda()
        
        paths = data['image_path']
        subs = data['participant_id']
        sess = data['session_id']

        label_list.extend(labels.cpu().numpy())
        path_list.extend(paths)
        sub_list.extend(subs)
        sess_list.extend(sess)

        outputs = model(images)
        prob = F.softmax(outputs, dim=1)
        prob_list.extend(prob.detach().cpu().numpy())
        
        outputs_max, _ = torch.max(outputs, dim=1)

    df = pd.DataFrame(columns=['participant_id', 'session_id', 'diagnosis', 'prob_AD'])
    
    # print(sub_list)
    # print(sess_list)
    # print(label_list)
    # print(np.stack(prob_list, axis=0)[:,1])
    df['img_path'] = path_list
    df['participant_id'] = sub_list
    df['session_id'] = sess_list
    df['diagnosis'] = label_list
    df['prob_AD'] = np.stack(prob_list, axis=0)[:,1]

    df.to_csv(join(save_dir,split+'_prediction_info.csv'), index=None)


def get_dataloader(split, batch_size, is_train=False, num_worker=4, transform=MinMaxNormalization(), idx_fold=0):
    dataset = MRIDataset('../data/ADNI_CAPS',split = split, transform=transform, idx_fold=idx_fold)
    dataloader = DataLoader(dataset,
                            shuffle=is_train,
                            batch_size=batch_size,
                            drop_last=is_train,
                            num_workers=num_worker,
                            pin_memory=False)
    return dataloader

if __name__ == "__main__":
    
    batch_size = 4
    save_dir = "prediction_2"
    makedirs(save_dir, exist_ok=True)

    config = config_utils.load("../mainScripts_test/policies/policy_2.yml")
    model = model_checkpoint("../results/checkpoints/seed_results/policy_lr1e-5_trial_1/checkpoint/epoch_0007.pth", None, config)
    model = model.cuda()

    dataloaders = {split:get_dataloader(split, batch_size=batch_size)
                   for split in ['train', 'test', 'val']}
    predict(model, dataloaders['train'], batch_size, save_dir, 'train')
    predict(model, dataloaders['test'], batch_size, save_dir, 'test')
    predict(model, dataloaders['val'], batch_size, save_dir, 'val')
    



