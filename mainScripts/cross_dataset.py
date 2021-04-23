import numpy as np
import pandas as pd
import math
import argparse

import tqdm
from easydict import EasyDict as edict

import torch
import torch.nn.functional as F

from model import get_model
from dataset import ValidDataset, MRIDataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from transform import MinMaxNormalization
from sklearn.metrics import precision_score, recall_score, accuracy_score, balanced_accuracy_score


def model_checkpoint(model, checkpoint_path):
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

    model.load_state_dict(checkpoint_dict)
    step = checkpoint['step'] if 'step' in checkpoint else -1
    last_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else -1

    return last_epoch, step 


def evaluate_single_dataset(model, dateset_name, dataloader, criterion, writer):
    model.eval()

    epoch = -1
    with torch.no_grad():
        batch_size = 12
        total_size = len(dataloader.dataset)
        total_step = math.ceil(total_size / batch_size)

        prediction_list = []
        label_list = []
        loss_list = []
        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
        for i, data in tbar:
            images = data['image'].cuda()
            labels = data['label'].cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            prediction_list.extend(predictions.cpu().numpy())
            label_list.extend(labels.cpu().numpy())
            loss_list.append(loss.item())

            f_epoch = epoch + i / total_step
            desc = '{:10s}'.format('{} test'.format(dateset_name))
            desc += ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
            tbar.set_description(desc)

        log_dict = {}
        labels = np.array(label_list)
        predictions = np.array(prediction_list)

        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        accuracy = accuracy_score(labels, predictions)
        ba = balanced_accuracy_score(labels, predictions)
        log_dict['acc'] = accuracy
        log_dict['ba'] = ba
        log_dict['pr'] = precision
        log_dict['rc'] = recall
        log_dict['loss'] = sum(loss_list) / len(loss_list)
        
        print(log_dict)
        for key, value in log_dict.items():
            if writer is not None:
                writer.add_scalar('{}/{}'.format(dateset_name, key), value, epoch)


def eval_cross_datasets(checkpoint_path):

    config = edict()
    config.model = edict()
    config.model.name = 'Conv5_FC3'
    config.model.params = None
    model = get_model(config).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    last_epoch, step = model_checkpoint(model, checkpoint_path)
    print('from checkpoint: {} last epoch:{}'.format(checkpoint_path, last_epoch))

    aibl_dataset = ValidDataset('AIBL_CAPS', 'AIBL_CAPS/aibl_info.csv', MinMaxNormalization())
    miriad_dataset = ValidDataset('MIRIAD_CAPS', 'MIRIAD_CAPS/miriad_test_info.csv', MinMaxNormalization())
    oasis3_dataset = ValidDataset('OASIS3_CAPS', 'OASIS3_CAPS/oasis3_test_info_2.csv', MinMaxNormalization())
    adni_dataset = MRIDataset('ADNI_CAPS', 'test', MinMaxNormalization())

    dataset_names = ['aibl', 'miriad', 'oasis3', 'adni'][::-1]
    datasets = [aibl_dataset, miriad_dataset, oasis3_dataset, adni_dataset][::-1]
    dataloaders = {name: DataLoader(dataset,
                                shuffle=False,
                                batch_size=12,
                                drop_last=False,
                                num_workers=6,
                                pin_memory=False)
                    for name, dataset in zip(dataset_names, datasets)
                } 


    writer = SummaryWriter('cross_dataset_test')
    for name in dataset_names:
        evaluate_single_dataset(model, name, dataloaders[name], criterion, writer)

    print("Finished cross dataset evaluation!")

def parse_args():
    parser = argparse.ArgumentParser(description='cross_datasets')
    parser.add_argument('--model', dest='model_path',
                        help='model path',
                        default=None, type=str)
    return parser.parse_args()

def main():
    print('cross datasets evaluation [AIBL/MIRIAD/OASIS3]')
    args = parse_args()
    if args.model_path is None:
      raise Exception('no model path')

    eval_cross_datasets(args.model_path)
    print('success!')

if __name__ == '__main__':
   main() 
