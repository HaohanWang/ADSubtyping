import torch
import pandas as pd
import numpy as np
from os.path import join
from copy import copy
from torch.utils.data import Dataset, sampler
from scipy.ndimage.filters import gaussian_filter

import itertools

import torch.utils.data
import torch.utils.data.sampler
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader


def get_dataset(config, split, transform=None, last_epoch=-1):
    f = globals().get(config.name)

    return f(config.dir,
             split=split,
             transform=transform,
             **config.params)


def get_dataloader(config, split, weighted_sampler=True, transform=None, **_):
    dataset = get_dataset(config.data, split, transform)

    is_train = 'train' == split
    batch_size = config.train.batch_size if is_train else config.eval.batch_size

    if is_train and weighted_sampler:
        train_label_list = np.array(dataset.label_list)
        class_sample_counts = [sum(train_label_list == label) for label in np.unique(train_label_list)]
        weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
        samples_weights = weights[train_label_list]
        sampler = WeightedRandomSampler(
                    weights=samples_weights,
                    num_samples=len(samples_weights),
                    replacement=True)
    else:
        sampler = None

    dataloader = DataLoader(dataset,
                            shuffle=(not is_train),
                            batch_size=batch_size,
                            drop_last=is_train,
                            sampler = sampler,
                            num_workers=config.transform.num_preprocessor,
                            pin_memory=False)
    return dataloader


class MRIDataset(Dataset):
    """Dataset of MRI organized in a CAPS folder."""

    def __init__(self,
                 img_dir,
                 split,
                 transform=None,
                 idx_fold=0,
                 num_fold=5,
                 split_prefix='split',
                 **_):
        """
        Args:
            img_dir (string): Directory of all the images.
            data_file (string): File name of the train/test split file.
            transform (callable, optional): Optional transform to be applied on a sample.

        """
        self.img_dir = img_dir
        self.split = split
        self.transform = transform
        self.idx_fold = idx_fold
        self.num_fold = num_fold
        self.split_prefix='split'
        df_path = join(img_dir, f'split.stratified.{idx_fold}.csv')
        df = pd.read_csv(df_path)
        df = df[df['split']==split]
        self.df = df.reset_index()
        self.diagnosis_code = {'CN': 0, 'AD': 1, 'MCI': 1, 'unlabeled': -1}
        self.label_list = [self.diagnosis_code[label] for label in self.df.diagnosis.values]
        self.size = self[0]['image'].numpy().size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'participant_id']
        sess_name = self.df.loc[idx, 'session_id']
        img_label = self.df.loc[idx, 'diagnosis']
        image_path = join(self.img_dir, 'subjects', img_name, sess_name, 'deeplearning_prepare_data', 'image_based', 't1_linear', img_name + '_' + sess_name + '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.pt')

        image = torch.load(image_path)
        label = self.diagnosis_code[img_label]

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label, 'participant_id': img_name, 'session_id': sess_name,
                  'image_path': image_path}

        return sample

    def session_restriction(self, session):
        """
            Allows to generate a new MRIDataset using some specific sessions only (mostly used for evaluation of test)

            :param session: (str) the session wanted. Must be 'all' or 'ses-MXX'
            :return: (DataFrame) the dataset with the wanted sessions
            """

        data_output = copy(self)
        if session == "all":
            return data_output
        else:
            df_session = self.df[self.df.session_id == session]
            df_session.reset_index(drop=True, inplace=True)
            data_output.df = df_session
            if len(data_output) == 0:
                raise Exception("The session %s doesn't exist for any of the subjects in the test data" % session)
            return data_output

class ValidDataset(Dataset):
    """Dataset of MRI organized in a CAPS folder."""

    def __init__(self,
                 img_dir,
                 data_file,
                 transform=None,
                 **_):
        """
        Args:
            img_dir (string): Directory of all the images.
            data_file (string): File name of the train/test split file.
            transform (callable, optional): Optional transform to be applied on a sample.

        """
        self.img_dir = img_dir
        self.data_file = data_file
        self.transform = transform
       
        self.df = pd.read_csv(self.data_file)
        self.diagnosis_code = {'CN': 0, 'AD': 1, 'MCI': 1, 'unlabeled': -1}

        self.size = self[0]['image'].numpy().size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'participant_id']
        sess_name = self.df.loc[idx, 'session_id']
        img_label = self.df.loc[idx, 'diagnosis']
        image_path = join(self.img_dir, 'subjects', img_name, sess_name, 't1_linear', img_name + '_' + sess_name + '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.pt')

        image = torch.load(image_path)
        label = self.diagnosis_code[img_label]

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label, 'participant_id': img_name, 'session_id': sess_name,
                  'image_path': image_path}

        return sample

    def session_restriction(self, session):
        """
            Allows to generate a new MRIDataset using some specific sessions only (mostly used for evaluation of test)

            :param session: (str) the session wanted. Must be 'all' or 'ses-MXX'
            :return: (DataFrame) the dataset with the wanted sessions
            """

        data_output = copy(self)
        if session == "all":
            return data_output
        else:
            df_session = self.df[self.df.session_id == session]
            df_session.reset_index(drop=True, inplace=True)
            data_output.df = df_session
            if len(data_output) == 0:
                raise Exception("The session %s doesn't exist for any of the subjects in the test data" % session)
            return data_output


def test():
    # dataset = MRIDataset('ADNI_CAPS', 'train', None)
    # print(len(dataset))
    dataset = MRIDataset('ADNI_CAPS', 'val', None)
    print(len(dataset))
    # dataset = MRIDataset('ADNI_CAPS', 'test', None)
    # print(len(dataset))
   #dataset = ValidDataset('AIBL_CAPS', 'AIBL_CAPS/aibl_info.csv', None)
   #dataset = ValidDataset('MIRIAD_CAPS', 'MIRIAD_CAPS/miriad_test_info.csv', None)
#   dataset = ValidDataset('OASIS3_CAPS', 'OASIS3_CAPS/oasis3_test_info.csv', None)
   #print(dataset[100])
   #  dataloader = DataLoader(dataset, shuffle=False, batch_size=4, drop_last=True, num_workers=2, pin_memory=False)
   #  for i, data in enumerate(dataloader):
   #      print('label', data['label'].cpu().numpy())
   #      print(data['label'])
   #      print('subs', data['participant_id'])
   #      print('sess', data['session_id'])
   #      if i == 0:

if __name__ == '__main__':
    test()
