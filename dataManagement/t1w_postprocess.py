import pandas as pd
from glob import glob

from os.path import join
from os import makedirs
import os
import re

import numpy as np 
import pandas as pd

import torch, os
import nibabel as nib

from tqdm import tqdm

def get_image_id(t1w_path):

    m1 = re.search(r'(_I[0-9]+_)', t1w_path)
    if m1 is None:
        raise ValueError('Input filename: {} can not find info.'.format(t1w_path))
    image_id = m1.group(1)
    return image_id


def save_as_pt(input_img, output_path):

    try:
        image_array = nib.load(input_img).get_fdata()
        image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()
        torch.save(image_tensor.clone(), output_path)
        return True
    except Exception as e:
        print(e)
        return False



df = pd.read_csv('adni1_anual2_1.5T.tsv', sep='\t')
filelist = glob('adni1_anual2_1.5T/*/*/*')
print('{} files to be postprocessed.'.format(len(filelist)))
out_dir = 'ADNI1_Annual_2_Yr_1.5T'
fail_ids = []
makedirs(out_dir, exist_ok=True)

visit_to_sess = {'Screening':'M00', 'Month 12':'M12', 'Month 24':'M24'}
sub_sess_diag = []

for i in tqdm(range(len(filelist))):
    image_id = int(get_image_id(filelist[i])[2:-1])
    entry = df[df['Image.ID']==image_id]
    if entry.shape[0] == 0:
        print('File: {} not in the list!'.format(filelist[i]))
    elif entry.shape[0] != 1:
        print('File: {} found multiple matches!'.format(filelist[i]))
    else:
        sub = 'sub-ADNI'+entry['PTID'].values[0].replace('_', '')
        visit = entry['Visit'].values[0]
        sess = 'ses-'+visit_to_sess[visit]
        diag = entry['DX'].values[0]
        sub_sess_diag.append( (sub, sess, diag) )

        sub_sess_dir = join(out_dir, sub, sess)
        makedirs(sub_sess_dir, exist_ok=True)
        sub_sess_img_path = join(sub_sess_dir, sub+'_'+sess+'_space-MNI_res-1x1x1.pt')
        if not save_as_pt(filelist[i], sub_sess_img_path):
            fail_ids.append(sub+'_'+sess)
            print(sub+'_'+sess)

sub_sess_diag_df = pd.DataFrame(sub_sess_diag, columns=['participant_id', 'session_id', 'diagnosis'])
sub_sess_diag_df.to_csv(join(out_dir, 'ADNI1_Annual_2_Yr_1.5T.tsv'), sep='\t', index=None)

