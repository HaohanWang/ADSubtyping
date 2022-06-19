__author__ = 'Haohan Wang'

import os


import numpy as np
from scipy import ndimage

from os.path import join
from matplotlib import pyplot as plt

import torch
import psutil


if psutil.Process().username() == 'haohanwang':
    BASE_DIR = '/media/haohanwang/Elements/'
else:
    BASE_DIR = '/home/ec2-user/mnt/home/ec2-user/alzstudy/AlzheimerData/'

def sub2adni(sub):
    return 'sub-ADNI' + ''.join(sub.split('_'))

def subsess2path(sub, sess):
    return join(BASE_DIR + 'ADNI_CAPS', 'subjects', sub, sess, 'deeplearning_prepare_data', 'image_based',
                't1_linear', \
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


def drawFromPath(sub, sess, filePath, i, kernelSize=4, zeroOut=0, threshold=0.05):
    # sub, sess = path2subsess(path)

    ex = imgfromsubsess(sub, sess)

    sagittal_idx = ex.shape[0] // 2
    coronal_idx = ex.shape[1] // 2
    axial_idx = ex.shape[2] // 2

    # sagittal_idx = ex.shape[0] - 100
    # coronal_idx = ex.shape[1] - 100
    # axial_idx = ex.shape[2] - 100

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

    #     im1 = ax[0].imshow(ex_sal[sagittal_idx, :, :], cmap=saliency_cmap, alpha=saliency_alpha)
    #     im2 = ax[1].imshow(ex_sal[:, coronal_idx, :], cmap=saliency_cmap, alpha=saliency_alpha)
    #     im3 = ax[2].imshow(ex_sal[:, :, axial_idx], cmap=saliency_cmap, alpha=saliency_alpha)
    saliencyDraw(ax[0], ex_sal[sagittal_idx, :, :], saliency_cmap, saliency_alpha)
    saliencyDraw(ax[1], ex_sal[:, coronal_idx, :], saliency_cmap, saliency_alpha)
    saliencyDraw(ax[2], ex_sal[:, :, axial_idx], saliency_cmap, saliency_alpha)
    # saliencyDraw(ax[0], ex_sal[sagittal_idx, :, :], view_cmap, saliency_alpha)
    # saliencyDraw(ax[1], ex_sal[:, coronal_idx, :], view_cmap, saliency_alpha)
    # saliencyDraw(ax[2], ex_sal[:, :, axial_idx], view_cmap, saliency_alpha)

    # os.makedirs(save_dir, exist_ok=True)
    # if order is None:
    #     fig.savefig(join(save_dir, '_'.join([label, sub, sess, '2d_view.png'])))
    #     fig.clf()
    # else:
    #     fig.savefig(join(save_dir, '_'.join([str(order), label, sub, sess, '2d_view.png'])))
    #     fig.clf()

    fig.savefig('imgs/idx_' + str(i) + '_ks_' + str(kernelSize) + '_th_' + str(threshold) +'_zo_' + str(zeroOut)  + '.png')

    # plt.show()


def getSubjSessList(filePath):
    subjLists = []
    sessLists = []

    for rl, dl, fl in os.walk(filePath + 'test/'):
        for d in dl:
            subjLists.append(d)
            for rl1, dl1, fl1 in os.walk(filePath + 'test/' + d + '/'):
                idx = np.random.choice(range(len(fl1)))
                sessLists.append(fl1[idx][:-4])

    print(subjLists)
    print(sessLists)

    return subjLists, sessLists


if __name__ == '__main__':

    # subjList, sessList = getSubjSessList(filePath)
    #
    # f = open('subjectList.txt', 'w')
    # for i in range(len(subjList)):
    #     f.writelines(subjList[i] + '\t' + sessList[i] + '\n')
    #
    # f.close()

    filePath = BASE_DIR + 'drop_block_after_flatten/'

    text = [line.strip() for line in open('subjectList.txt')]
    subjList = []
    sessList = []
    for line in text:
        items = line.split('\t')
        subjList.append(items[0])
        sessList.append(items[1])

    # kernelSize = 0
    # zeroOut = 0
    # threshold = 0.05
    #
    # for i in range(len(sessList)):
    #     subj = subjList[i]
    #     sess = sessList[i]
    #
    #     drawFromPath(subj, sess, filePath, i, kernelSize, zeroOut, threshold)

    for kernelSize in [4]:
        for zeroOut in [0]:
            for threshold in [0.05]:
                for i in range(len(sessList)):
                    subj = subjList[i]
                    sess = sessList[i]

                    drawFromPath(subj, sess, filePath, i, kernelSize, zeroOut, threshold)

                    if i > 30:
                        break

