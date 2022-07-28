__author__ = 'Haohan Wang'

import numpy as np
import pandas as pd
import torch

import re
import os
from os.path import join, dirname, basename
from tqdm import tqdm
from glob import glob

# import nibabel as nib
from scipy import ndimage

import matplotlib.pyplot as plt
import psutil

if psutil.Process().username() == 'haohanwang':
    BASE_DIR = '/media/haohanwang/Elements/'
else:
    BASE_DIR = '/mnt/home/ec2-user/alzstudy/AlzheimerData/'


def sub2adni(sub):
    return 'sub-ADNI' + ''.join(sub.split('_'))


def df2path(df, i):
    # return join('/media/haohanwang/Elements/saliency_map_dropblock2', df.split.iloc[i], df.participant_id.iloc[i], df.session_id.iloc[i] + '.npy')
    # return join(BASE_DIR + 'saliency_map_torch', df.split.iloc[i], df.participant_id.iloc[i],
    #             df.session_id.iloc[i] + '.npy')
    return join(BASE_DIR + 'mci_finetune_clean_saliency_e_88', df.split.iloc[i], df.participant_id.iloc[i],
                df.session_id.iloc[i] + '.npy')

def path2subsess(path):
    sub = re.search(r'sub-[a-zA-Z0-9]{1,}', path).group(0)
    sess = re.search(r'ses-M[0-9]{1,}', path).group(0)
    return sub, sess


def subsess2path(sub, sess):
    return join(BASE_DIR + 'ADNI_CAPS', 'subjects', sub, sess, 'deeplearning_prepare_data', 'image_based', 't1_linear', \
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



def drawFromPath_zero_out(sub, sess, filePath, i, kernelSize=4, zeroOut=0, threshold=0.05):
    ex = imgfromsubsess(sub, sess)

    sagittal_idx = ex.shape[0] // 2
    coronal_idx = ex.shape[1] // 2
    axial_idx = ex.shape[2] // 2

    view_cmap = 'gray'
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
    ax = axes.ravel()
    ax[0].imshow(ex[sagittal_idx, :, :], cmap=view_cmap)
    ax[1].imshow(ex[:, coronal_idx, :], cmap=view_cmap)
    ax[2].imshow(ex[:, :, axial_idx], cmap=view_cmap)
    ax[0].set_title('Sagittal view')
    ax[1].set_title('Coronal view')
    ax[2].set_title('Axial view')

    saliency_cmap = 'jet'
    saliency_alpha = 0.3

    try:
        ex_sal = np.load(filePath + 'test/' + sub + '/' + sess + '.npy')
    except FileNotFoundError:
        print("Subject example not found in {}".format(filePath))
        return

    if zeroOut == 1:
        sal_mean = np.mean(ex_sal)
        sal_std = np.std(ex_sal)

        ex_sal = np.where(ex_sal < sal_mean - 0. * sal_std, 0, ex_sal)
        # ex_sal = (ex_sal - np.min(ex_sal)) / (np.max(ex_sal) - np.min(ex_sal))


    ex_sal = (ex_sal - np.min(ex_sal)) / (np.max(ex_sal) - np.min(ex_sal))
    ex_sal = np.where(ex_sal < threshold, 0, ex_sal)

    # smoothing over neighboring pixels
    if kernelSize != 0:
        kernel = np.ones([kernelSize, kernelSize, kernelSize]) / (kernelSize * kernelSize * kernelSize)
        ex_sal = ndimage.convolve(ex_sal, kernel, mode='constant', cval=0.0)
    # ex_sal = (ex_sal - np.min(ex_sal))/(np.max(ex_sal) - np.min(ex_sal))

    if zeroOut == 2:
        sal_mean = np.mean(ex_sal)
        sal_std = np.std(ex_sal)
        # zero-out small, insignificant saliency / gradients

        ex_sal = np.where(ex_sal < sal_mean - 0. * sal_std, 0, ex_sal)
        # ex_sal = (ex_sal - np.min(ex_sal)) / (np.max(ex_sal) - np.min(ex_sal))

    saliencyDraw(ax[0], ex_sal[sagittal_idx, :, :], saliency_cmap, saliency_alpha)
    saliencyDraw(ax[1], ex_sal[:, coronal_idx, :], saliency_cmap, saliency_alpha)
    saliencyDraw(ax[2], ex_sal[:, :, axial_idx], saliency_cmap, saliency_alpha)

    # os.makedirs(save_dir, exist_ok=True)
    # if order is None:
    #     fig.savefig(join(save_dir, '_'.join([label, sub, sess, '2d_view.png'])))
    #     fig.clf()
    # else:
    #     fig.savefig(join(save_dir, '_'.join([str(order), label, sub, sess, '2d_view.png'])))
    #     fig.clf()

    fig.savefig('imgs/idx_' + str(i) + '_ks_' + str(kernelSize) + '_th_' + str(threshold) +'_zo_' + str(zeroOut) + '.png')



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

    try:
        ex_sal = np.load(path)
    except FileNotFoundError:
        print("Subject example not found in {}".format(path))
        return

    ex_sal = (ex_sal - np.min(ex_sal)) / (np.max(ex_sal) - np.min(ex_sal))
    ex_sal = np.where(ex_sal < 0.05, 0, ex_sal)

    # smoothing over neighboring pixels
    kernel = np.ones([4, 4, 4]) / (4*4*4)
    ex_sal = ndimage.convolve(ex_sal, kernel, mode='constant', cval=0.0)

    # sal_mean = np.mean(ex_sal)
    # sal_std = np.std(ex_sal)

    # zero-out small, insignificant saliency / gradients
    ex_sal = np.where(ex_sal < 0.05, 0, ex_sal)

    #     im1 = ax[0].imshow(ex_sal[sagittal_idx, :, :], cmap=saliency_cmap, alpha=saliency_alpha)
    #     im2 = ax[1].imshow(ex_sal[:, coronal_idx, :], cmap=saliency_cmap, alpha=saliency_alpha)
    #     im3 = ax[2].imshow(ex_sal[:, :, axial_idx], cmap=saliency_cmap, alpha=saliency_alpha)
    saliencyDraw(ax[0], ex_sal[sagittal_idx, :, :], saliency_cmap, saliency_alpha)
    saliencyDraw(ax[1], ex_sal[:, coronal_idx, :], saliency_cmap, saliency_alpha)
    saliencyDraw(ax[2], ex_sal[:, :, axial_idx], saliency_cmap, saliency_alpha)
    # saliencyDraw(ax[0], ex_sal[sagittal_idx, :, :], view_cmap, saliency_alpha)
    # saliencyDraw(ax[1], ex_sal[:, coronal_idx, :], view_cmap, saliency_alpha)
    # saliencyDraw(ax[2], ex_sal[:, :, axial_idx], view_cmap, saliency_alpha)

    os.makedirs(save_dir, exist_ok=True)
    if order is None:
        fig.savefig(join(save_dir, '_'.join([label, sub, sess, '2d_view.png'])))
        fig.clf()
    else:
        fig.savefig(join(save_dir, '_'.join([str(order), label, sub, sess, '2d_view.png'])))
        fig.clf()

if __name__ == '__main__':
    subs = glob(BASE_DIR + 'ADNI_CAPS/subjects/*')

    print(subs[:5])

    ex = imgfromsubsess('sub-ADNI094S0711', 'ses-M00')
    plt.imshow(ex[:,:,90], cmap='gray')

    snp_subs = np.loadtxt(BASE_DIR + 'SNP/samples.txt', dtype=str)
    snp_subs = np.array(list(map(sub2adni, snp_subs)))

    df = pd.read_csv(BASE_DIR +'ADNI_CAPS/mci_finetune_clean.csv')
    tmp = df[df['participant_id'].isin(snp_subs)]
    tmp = tmp.sort_values(by=['session_id'], ascending=False)
    tmp = tmp.drop_duplicates(subset=['participant_id'])
    df_snp = tmp.reset_index(drop=True)

    train_df = pd.read_csv(BASE_DIR + 'mci_finetune_clean_saliency_e_88/train_saliency_info.csv')
    test_df = pd.read_csv(BASE_DIR + 'mci_finetune_clean_saliency_e_88/test_saliency_info.csv')
    val_df = pd.read_csv(BASE_DIR + 'mci_finetune_clean_saliency_e_88/val_saliency_info.csv')
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
        print(p)
        print(l)
        drawFromPath(p,l, BASE_DIR + 'mci_visualization_finetune_clean_0727_e88')

