import torch

import os
import numpy as np
from glob import glob
import traceback

from multiprocessing.pool import Pool
from tqdm import tqdm

def save_as_pt(input_img):
    """
    This function is to transfer nii.gz file into .pt format, in order to train the classifiers model more efficient when loading the data.
    :param input_img:
    :return:
    """

    import torch, os
    import nibabel as nib

    try:
        output_file = input_img.split('.nii.gz')[0] + '.pt'
        if os.path.exists(output_file):
            return
        image_array = nib.load(input_img).get_fdata()
        image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()
    ## make sure the tensor dtype is torch.float32
    # save
        torch.save(image_tensor.clone(), output_file)
    except:
        pass


def save_as_pts(pid, input_imgs):
    for input_img in tqdm(input_imgs, postfix=pid):
        save_as_pt(input_img) 

def main():
    num_workers = 7
    img_list = glob('OASIS3_CAPS/subjects/sub*/ses*/t1_linear/*desc-Crop_res-1x1x1_T1w.nii.gz')
    img_splits = np.array_split(img_list, num_workers)
    assert sum([len(v) for v in img_splits]) == len(img_list)
    p = Pool(num_workers)
    for i, split in enumerate(img_splits):
        p.apply_async(
            save_as_pts, args=(str(i), list(split))
        )
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')

if __name__ == "__main__":
    main()
