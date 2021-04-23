__author__ = 'Haohan Wang'

import numpy as np
import pandas as pd
import torch

import re
import os
from os.path import join, dirname, basename
from tqdm import tqdm
from glob import glob

import nibabel as nib

import matplotlib.pyplot as plt


def sub2adni(sub):
    return 'sub-ADNI' + ''.join(sub.split('_'))


def df2path(df, i):
    return join('/media/haohanwang/Elements/saliency_map_dropblock2', df.split.iloc[i], df.participant_id.iloc[i], df.session_id.iloc[i] + '.npy')


def path2subsess(path):
    sub = re.search(r'sub-[a-zA-Z0-9]{1,}', path).group(0)
    sess = re.search(r'ses-M[0-9]{1,}', path).group(0)
    return sub, sess


def subsess2path(sub, sess):
    return join('/media/haohanwang/Elements/ADNI_CAPS', 'subjects', sub, sess, 'deeplearning_prepare_data', 'image_based', 't1_linear', \
                sub + '_' + sess + '_' + 'T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.pt')


def imgfromsubsess(sub, sess):
    return torch.load(subsess2path(sub, sess)).squeeze(0).numpy()



def saliencyDraw(ax, ex_sal, saliency_cmap, saliency_alpha):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)

    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)

    im = ax.imshow(ex_sal, cmap=saliency_cmap, alpha=saliency_alpha)

    plt.colorbar(im, cax=ax_cb)
    ax_cb.yaxis.tick_right()
    ax_cb.yaxis.set_tick_params(labelright=False)


def drawFromPath(path, label, save_dir='2d_view', order=None):
    sub, sess = path2subsess(path)

    ex = imgfromsubsess(sub, sess)
    sagittal_idx, coronal_idx, axial_idx = [i//2 for i in ex.shape]
    view_cmap = 'gray'
    fig, axes = plt.subplots(ncols=3, figsize=(15,5))
    ax = axes.ravel()
    ax[0].imshow(ex[sagittal_idx, :, :], cmap=view_cmap)
    ax[1].imshow(ex[:, coronal_idx, :], cmap=view_cmap)
    ax[2].imshow(ex[:, :, axial_idx], cmap=view_cmap)
    ax[0].set_title('Sagittal view')
    ax[1].set_title('Coronal view')
    ax[2].set_title('Axial view')

    saliency_cmap = 'jet'
    saliency_alpha = 0.3
    ex_sal = np.load(path)
    #     im1 = ax[0].imshow(ex_sal[sagittal_idx, :, :], cmap=saliency_cmap, alpha=saliency_alpha)
    #     im2 = ax[1].imshow(ex_sal[:, coronal_idx, :], cmap=saliency_cmap, alpha=saliency_alpha)
    #     im3 = ax[2].imshow(ex_sal[:, :, axial_idx], cmap=saliency_cmap, alpha=saliency_alpha)
    saliencyDraw(ax[0], ex_sal[sagittal_idx, :, :], saliency_cmap, saliency_alpha)
    saliencyDraw(ax[1], ex_sal[:, coronal_idx, :], saliency_cmap, saliency_alpha)
    saliencyDraw(ax[2], ex_sal[:, :, axial_idx], saliency_cmap, saliency_alpha)

    # os.makedirs(save_dir)
    if order is None:
        fig.savefig(join(save_dir, '_'.join([label, sub, sess, '2d_view.png'])))
        fig.clf()
    else:
        fig.savefig(join(save_dir, '_'.join([str(order), label, sub, sess, '2d_view.png'])))
        fig.clf()

if __name__ == '__main__':
    subs = glob('/media/haohanwang/Elements/ADNI_CAPS/subjects/*')

    print subs[:5]

    ex = imgfromsubsess('sub-ADNI094S0711', 'ses-M00')
    plt.imshow(ex[:,:,90], cmap='gray')

    snp_subs = np.loadtxt('SNP/samples.txt', dtype=str)
    snp_subs = np.array(list(map(sub2adni, snp_subs)))

    df = pd.read_csv('/media/haohanwang/Elements/ADNI_CAPS/split.stratified.0.csv')
    tmp = df[df['participant_id'].isin(snp_subs)]
    tmp = tmp.sort_values(by=['session_id'], ascending=False)
    tmp = tmp.drop_duplicates(subset=['participant_id'])
    df_snp = tmp.reset_index(drop=True)

    train_df = pd.read_csv('/media/haohanwang/Elements/saliency_map_dropblock2/train_sailency_info.csv')
    test_df = pd.read_csv('/media/haohanwang/Elements/saliency_map_dropblock2/test_sailency_info.csv')
    val_df = pd.read_csv('/media/haohanwang/Elements/saliency_map_dropblock2/val_sailency_info.csv')
    df_pred = pd.concat([train_df, test_df, val_df])
    df_pred = df_pred.reset_index(drop=True)

    df_snp = df_snp.merge(df_pred, how='inner', on=['participant_id', 'session_id'])
    df_snp['diagnosis_pred'] = df_snp['prob_AD'].map(lambda x: np.round(x))
    snp_correct = df_snp['diagnosis_y'] == df_snp['diagnosis_pred'].map(lambda x: int(x))
    df_snp_correct = df_snp[snp_correct]

    snp_paths = [df2path(df_snp, i) for i in range(len(df_snp))]
    snp_labels = df_snp.diagnosis_x.values
    snp_correct_paths = [df2path(df_snp_correct, i) for i in range(len(df_snp_correct))]
    snp_correct_labels = df_snp_correct.diagnosis_x.values

    for p, l in tqdm(zip(snp_correct_paths, snp_correct_labels)):
        print p
        print l
        drawFromPath(p,l,'2d_view_dropblock2')


