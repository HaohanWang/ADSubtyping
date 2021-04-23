from google.cloud import storage
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import torchvision.transforms as T

from os.path import join


class MRIDataset(Dataset):
    """Dataset of MRI organized in a CAPS folder."""
    def __init__(self, caps_dir, idx_fold, num_fold, split, transform=None):

        self.caps_dir = caps_dir
        self.idx_fold = idx_fold
        self.num_fold = num_fold
        self.df = pd.read_csv(
            join(caps_dir, f'split.stratified.{idx_fold}.csv'))
        self.df = self.df[self.df['split'] == split]
        self.df = self.df.reset_index()
        self.transform = transform
        self.diagnosis_code = {'CN': 0, 'AD': 1, 'MCI': 1, 'unlabeled': -1}
        self.label_list = [
            self.diagnosis_code[label] for label in self.df.diagnosis.values
        ]
        self.size = self[0]['image'].numpy().size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'participant_id']
        sess_name = self.df.loc[idx, 'session_id']
        img_label = self.df.loc[idx, 'diagnosis']
        image_path = join(
            self.caps_dir, 'subjects', img_name, sess_name,
            'deeplearning_prepare_data', 'image_based', 't1_linear',
            img_name + '_' + sess_name +
            '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.pt')

        image = torch.load(image_path)
        label = self.diagnosis_code[img_label]

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label}
        return sample


class MinMaxNormalization(object):
    """Normalizes a tensor between 0 and 1"""
    def __call__(self, image):
        return (image - image.min()) / (image.max() - image.min())


def weightedDataloader(dataset, batch_size, num_workers):
    label_list = np.array(dataset.label_list)
    class_sample_counts = [
        sum(label_list == label) for label in np.unique(label_list)
    ]
    weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
    samples_weights = weights[label_list]
    sampler = WeightedRandomSampler(weights=samples_weights,
                                    num_samples=len(samples_weights),
                                    replacement=True)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            num_workers=num_workers)

    return dataloader


def load_data(caps_dir, seed, num_fold, batch_size, num_workers, noise):
    """Loads the data"""
    torch.manual_seed(seed)

    # add random noise during training
    train_aug = T.Compose([
        MinMaxNormalization(),
        T.Lambda(lambda x: x + noise * torch.randn_like(x)),
        MinMaxNormalization()
    ])
    val_aug = MinMaxNormalization()

    train_datasets = [
        MRIDataset(caps_dir,
                   idx_fold,
                   num_fold,
                   split='train',
                   transform=train_aug) for idx_fold in range(num_fold)
    ]
    val_datasets = [
        MRIDataset(caps_dir,
                   idx_fold,
                   num_fold,
                   split='val',
                   transform=val_aug) for idx_fold in range(num_fold)
    ]

    # use weighted sampler in train_loaders

    train_loaders = [
        weightedDataloader(train_datasets[idx_fold], batch_size, num_workers)
        for idx_fold in range(num_fold)
    ]
    val_loaders = [
        DataLoader(val_datasets[idx_fold],
                   batch_size=batch_size,
                   num_workers=num_workers,
                   shuffle=True) for idx_fold in range(num_fold)
    ]

    return train_loaders, val_loaders


def save_model(job_dir, model_name):
    """Saves the model to Google Cloud Storage"""
    # Example: job_dir = 'gs://BUCKET_ID/hptuning_sonar/1'
    job_dir = job_dir.replace('gs://', '')  # Remove the 'gs://'
    # Get the Bucket Id
    bucket_id = job_dir.split('/')[0]
    # Get the path. Example: 'hptuning_sonar/1'
    bucket_path = job_dir.lstrip('{}/'.format(bucket_id))

    # Upload the model to GCS
    bucket = storage.Client().bucket(bucket_id)
    blob = bucket.blob('{}/{}'.format(bucket_path, model_name))
    blob.upload_from_filename(model_name)


def save_checkpoint(args, model, optimizer, epoch, fold, step, weights_dict=None, name=None):
  checkpoint_dir = join(args.job_dir, 'checkpoint')

  if name:
    checkpoint_path = join(checkpoint_dir, '{}.pth'.format(name))
  else:
    checkpoint_path = join(checkpoint_dir, 'epoch_{:04d}_fold_{:02d}.pth'.format(epoch, fold))

  if weights_dict is None:
    weights_dict = {
      'state_dict': model.state_dict(),
      'optimizer_dict' : optimizer.state_dict(),
      'epoch' : epoch,
      'step' : step,
    }
  torch.save(weights_dict, checkpoint_path)

def test():
    import warnings
    warnings.filterwarnings("ignore")
    # test MRIDataset
    # dataset = MRIDataset('ADNI_CAPS', 0, 5, 'train', None)
    # print(len(dataset))
    # dataset = MRIDataset('ADNI_CAPS', 0, 5, 'val', None)
    # print(len(dataset))
    # dataset = MRIDataset('ADNI_CAPS', 0, 5, 'test', None)
    # print(len(dataset))

    # test
    train_loaders, val_loaders = load_data('ADNI_CAPS', 0, 5,
                                            1, 4, 0)
    print(len(train_loaders[0]))
    print(len(val_loaders[0]))
    # image = next(iter(train_loaders[0]))["image"]
    # print(image.min(), image.max())
    # print(image)


if __name__ == '__main__':
    test()