from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../')

import os
import math
import argparse
import pprint
import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from tensorboardX import SummaryWriter

from dataMangement.dataset import get_dataloader
from dataMangement.transform import get_transform, MinMaxNormalization
from model import get_model
from optimizer import get_optimizer

from utility import utils
from utility import config_utils
from utility import checkpoint_utils

from sklearn.metrics import precision_score, recall_score, accuracy_score, balanced_accuracy_score
import torchattacks

torch.manual_seed(1)
np.random.seed(1)

def get_statics(outputs, labels, log_dict):
    predictions = torch.argmax(outputs, dim=1).detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    accuracy = accuracy_score(labels, predictions)
    ba = balanced_accuracy_score(labels, predictions)

    log_dict['acc'] = accuracy
    log_dict['ba'] = ba
    log_dict['pr'] = precision
    log_dict['rc'] = recall


def evaluate_single_epoch(config, model, dataloader, criterion,
                          epoch, writer, postfix_dict):
    model.eval()

    with torch.no_grad():
        batch_size = config.eval.batch_size
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
            predictions = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            prediction_list.extend(predictions)
            label_list.extend(labels)
            loss_list.append(loss.item())

            f_epoch = epoch + i / total_step
            desc = '{:5s}'.format('val')
            desc += ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
            tbar.set_description(desc)
            tbar.set_postfix(**postfix_dict)

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

        for key, value in log_dict.items():
            if writer is not None:
                writer.add_scalar('val/{}'.format(key), value, epoch)
            postfix_dict['val/{}'.format(key)] = value

        return ba


def train_single_epoch(config, model, dataloader, criterion, optimizer, pgd_attack,
                       epoch, writer, postfix_dict):
    model.train()

    batch_size = config.train.batch_size
    total_size = len(dataloader.dataset)
    total_step = math.ceil(total_size / batch_size)

    log_dict = {}
    tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
    for i, data in tbar:
        images = data['image'].cuda()
        # pgd_images = pgd_attack(data['image'], data['label']).cuda()
        labels = data['label'].cuda()
        outputs = model(images)
        # pgd_outputs = model(pgd_images)
        # loss = criterion(outputs, labels) + 0.01*torch.norm(outputs-pgd_outputs,dim=1).pow(2).sum()
        loss = criterion(outputs, labels)
        log_dict['loss'] = loss.item()

        # probabilities = F.softmax(outputs, dim=1)
        # predictions = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        # labels = labels.detach().cpu().numpy()
        # precision = precision_score(labels, predictions)
        # recall = recall_score(labels, predictions)
        # accuracy = accuracy_score(labels, predictions)
        # ba = balanced_accuracy_score(labels, predictions)

        # log_dict['acc'] = accuracy
        # log_dict['ba'] = ba
        # log_dict['pr'] = precision
        # log_dict['rc'] = recall
        get_statics(outputs, labels, log_dict)

        loss.backward()

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
    num_epochs = config.train.num_epochs

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.cuda()
    model = model.load_state_dict('../pretrainModels/best_model/fold_0/model_best.pth.tar')

    postfix_dict = {'train/lr': 0.0,
                    'train/loss': 0.0,
                    'train/acc': 0.0,
                    'train/ba': 0.0,
                    'train/pr': 0.0,
                    'train/rc': 0.0,
                    'val/loss': 0.0,
                    'val/acc': 0.0,
                    'val/ba': 0.0,
                    'val/pr': 0.0,
                    'val/rc': 0.0}

    ba_list = []
    best_ba = 0.0
    best_ba_mavg = 0.0
    for epoch in range(start_epoch, num_epochs):
        torch.cuda.empty_cache()
        # train phase
        train_single_epoch(config, model, dataloaders['train'],
                           criterion, optimizer, pgd_attack, epoch, writer, postfix_dict)

        # val phase
        ba = evaluate_single_epoch(config, model, dataloaders['val'],
                                   criterion, epoch, writer, postfix_dict)

        checkpoint_utils.save_checkpoint(config, model, optimizer, epoch, 0)

        ba_list.append(ba)
        ba_list = ba_list[-10:]
        ba_mavg = sum(ba_list) / len(ba_list)

        if ba > best_ba:
            best_ba = ba
        if ba_mavg > best_ba_mavg:
            best_ba_mavg = ba_mavg
    return {'ba': best_ba, 'ba_mavg': best_ba_mavg}


def run(config):
    train_dir = config.train.dir

    model = get_model(config).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(config, model.parameters())
    pgd_attack = torchattacks.PGD(model, eps=config.pgd.eps, alpha=config.pgd.alpha, steps=config.pgd.iters, random_start=True)

    checkpoint = checkpoint_utils.get_initial_checkpoint(config)
    if checkpoint is not None:
        last_epoch, step = checkpoint_utils.load_checkpoint(model, optimizer, checkpoint)
    else:
        last_epoch, step = -1, -1

    print('from checkpoint: {} last epoch:{}'.format(checkpoint, last_epoch))

    train_aug = T.Compose([MinMaxNormalization(), T.Lambda(lambda x : x + config.pgd.eps/10*torch.randn_like(x)), MinMaxNormalization()])
    dataloaders = {}
    dataloaders['train'] = get_dataloader(config, 'train', train_aug)
    dataloaders['val'] = get_dataloader(config, 'val', MinMaxNormalization())

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

    print('train ADNI1/ADNIGO/ADNI2 1.5T MRI images.')
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
