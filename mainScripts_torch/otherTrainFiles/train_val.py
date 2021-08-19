from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import argparse
import pprint
import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from dataset import get_dataloader
from transform import get_transform, MinMaxNormalization
from model import get_model
from optimizer import get_optimizer

import utils
import config_utils
import checkpoint_utils

from sklearn.metrics import precision_score, recall_score, accuracy_score, balanced_accuracy_score
import torchattacks



def evaluate_single_epoch(config, model, dataloader):
    model.eval()

    with torch.no_grad():
        batch_size = config.eval.batch_size
        total_size = len(dataloader.dataset)
        total_step = math.ceil(total_size / batch_size)

        
        label_list = []
        img_path_list = []
        prob_diff_list = []
        prob_list = []

        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
        for i, data in tbar:
            images = data['image']
            labels = data['label'].cpu().numpy()
            img_path = data['image_path']
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()
            predictions = np.argmax(probabilities, axis=1)
            prob_diff = np.abs(probabilities[:,0]-probabilities[:,1])
            prob_diff = prob_diff * (predictions!=labels)

            label_list.extend(list(labels))

            img_path_list.extend(img_path)
            prob_diff_list.extend(list(prob_diff))
            prob_list.append(probabilities)

        prob_diff_list = np.array(prob_diff_list)
        prob_list = np.vstack(prob_list)
        idx = np.argmax(prob_diff_list)
        print(img_path_list[idx])
        print(label_list[idx])
        print(prob_list[idx])
        print(prob_diff_list[idx])


def train_single_epoch(config, model, dataloader, criterion, optimizer, pgd_attack,
                       epoch, writer, postfix_dict):
    model.train()

    batch_size = config.train.batch_size
    total_size = len(dataloader.dataset)
    total_step = math.ceil(total_size / batch_size)

    log_dict = {}
    tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
    for i, data in tbar:
        # images = data['image'].cpu()
        images = pgd_attack(data['image'], data['label']).cpu()
        labels = data['label'].cpu()
        outputs = model(images)
        loss = criterion(outputs, labels)
        log_dict['loss'] = loss.item()

        probabilities = F.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        accuracy = accuracy_score(labels, predictions)
        ba = balanced_accuracy_score(labels, predictions)

        log_dict['acc'] = accuracy
        log_dict['ba'] = ba
        log_dict['pr'] = precision
        log_dict['rc'] = recall

        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if config.train.num_grad_acc is None:
            optimizer.step()
            optimizer.zero_grad()
        elif (i+1) % config.train.num_grad_acc == 0:
            optimizer.step()
            optimizer.zero_grad()

        f_epoch = epoch + i / total_step

        log_dict['lr'] = optimizer.param_groups[0]['lr']
        for key, value in log_dict.items():
            postfix_dict['train/{}'.format(key)] = value

        desc = '{:5s}'.format('train')
        desc += ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
        tbar.set_description(desc)
        tbar.set_postfix(**postfix_dict)

        if i % 5 == 0:
            log_step = int(f_epoch * 1000)
            if writer is not None:
                for key, value in log_dict.items():
                    writer.add_scalar('train/{}'.format(key), value, log_step)


def train(config, model, dataloaders, criterion, optimizer, pgd_attack, writer, start_epoch):

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.cpu()

    evaluate_single_epoch(config, model, dataloaders['val'])

    torch.cpu.empty_cache()


def run(config):
    train_dir = config.train.dir

    model = get_model(config).cpu()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(config, model.parameters())
    pgd_attack = torchattacks.PGD(model, eps=config.pgd.eps, alpha=config.pgd.alpha/255, iters=config.pgd.iters)

    checkpoint = checkpoint_utils.get_initial_checkpoint(config)
    if checkpoint is not None:
        last_epoch, step = checkpoint_utils.load_checkpoint(model, optimizer, checkpoint)
    else:
        last_epoch, step = -1, -1
    print('from checkpoint: {} last epoch:{}'.format(checkpoint, last_epoch))

    dataloaders = {split:get_dataloader(config, split, MinMaxNormalization())
                   for split in ['train', 'val']}

    writer = SummaryWriter(config.train.dir)
    ba_stat  = train(config, model, dataloaders, criterion, optimizer, pgd_attack,
          writer, last_epoch+1)
    print(ba_stat)


def parse_args():
    parser = argparse.ArgumentParser(description='ADNI')
    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default=None, type=str)
    return parser.parse_args()


def main():
    import warnings
    warnings.filterwarnings("ignore")

    print('train ADNI1 Annual 2 Yr 1.5T MRI images.')
    args = parse_args()
    if args.config_file is None:
      raise Exception('no configuration file')

    config = config_utils.load(args.config_file)
    pprint.PrettyPrinter(indent=2).pprint(config)
    utils.prepare_train_directories(config)
    run(config)

    print('success!')


if __name__ == '__main__':
    main()
