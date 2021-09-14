#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import pandas as pd
from os.path import join, dirname, basename
from tqdm import tqdm
import os
import torch

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# In[31]:


import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

from scipy.cluster.hierarchy import dendrogram, linkage



def sub2adni(sub):
    return 'sub-ADNI'+''.join(sub.split('_'))

# In[3]:

subs = np.loadtxt('SNP/samples.txt', dtype=str)
subs = np.array(list(map(sub2adni, subs)))


# In[4]:

df = pd.read_csv('/media/haohanwang/Elements/ADNI_CAPS/split.stratified.0.csv')
tmp = df[df['participant_id'].isin(subs)]
tmp = tmp.sort_values(by=['session_id'], ascending=False)
tmp = tmp.drop_duplicates(subset=['participant_id'])
df_snp = tmp.reset_index(drop=True)


# In[5]:


def df2path(df, i):
    return join('/media/haohanwang/Elements/saliency_map2', df.split.iloc[i], df.participant_id.iloc[i], df.session_id.iloc[i]+'.npy')


# In[6]:


paths = [df2path(df_snp, i) for i in range(len(df_snp))]
# np.save('snp_paths.npy', paths)


# In[7]:


train_df = pd.read_csv('/media/haohanwang/Elements/saliency_map/train_sailency_info.csv')
test_df = pd.read_csv('/media/haohanwang/Elements/saliency_map/test_sailency_info.csv')
val_df = pd.read_csv('/media/haohanwang/Elements/saliency_map/val_sailency_info.csv')
df_pred = pd.concat([train_df, test_df, val_df])
df_pred = df_pred.reset_index(drop=True)


# In[8]:


df_snp = df_snp.merge(df_pred, how='inner', on=['participant_id', 'session_id'])
df_snp['diagnosis_pred'] = df_snp['prob_AD'].map(lambda x: np.round(x))

snp_correct = df_snp['diagnosis_y'] == df_snp['diagnosis_pred'].map(lambda x: int(x))
df_snp_correct = df_snp[snp_correct]


# In[9]:


correct_paths = [df2path(df_snp_correct, i) for i in range(len(df_snp_correct))]


def raw_dist_clustering():

    # calculating the distances

    n = len(df_snp_correct)
    dist = np.zeros((n, n))
    for i in tqdm(range(n)):
        vi = np.load(correct_paths[i]).flatten()
        for j in range(i+1, n):
            vj = np.load(correct_paths[j]).flatten()
            dist[i][j] = np.linalg.norm(vi-vj)

    np.save('raw_dist.npy', dist)

    # clustering
    labels = [df_snp_correct.participant_id.iloc[i]+'\t\t'+df_snp_correct.diagnosis_x.iloc[i] for i in range(len(df_snp_correct))]
    d = []
    for l in labels:
        if l.split('\t')[-1] == 'CN':
            d.append(0)
        else:
            d.append(1)
    d = np.array(d)
    print (len(labels), np.mean(d))

    dist = np.load('raw_dist.npy')

    # tsne_em = TSNE(n_components=2, perplexity=30.0, n_iter=1000, verbose=1).fit_transform(dist)
    pca_em = PCA(n_components=2).fit_transform(dist)
    plt.scatter(pca_em[d==0,0], pca_em[d==0,1], color='blue', alpha=0.5)
    plt.scatter(pca_em[d==1,0], pca_em[d==1,1], color='red', alpha=0.5)
    plt.show()

    i, j = np.triu_indices(dist.shape[0], k=1)
    X = dist[i, j]
    Z = linkage(X, 'ward')
    fig = plt.figure(figsize=(50, 25))
    dn = dendrogram(Z, labels=labels, leaf_font_size=15)

    # # write out clustering results
    nameWrite = open('raw_labels.txt', 'w')
    for name in dn['ivl']:
        nameWrite.writelines(name.split('\t')[0] + '\n')
    nameWrite.close()



    # visualize the clustering results
    fig.suptitle('raw dist', fontsize=16)
    fig.savefig('raw.pdf')


def thresholding(x, r):
    thres = np.quantile(x, r)
    return x*(x>thres)

def thresholding_dist_clustering():
    # calculate distance
    # n = len(df_snp_correct)
    # thres_dist = np.zeros((n, n))
    # r = 0.95
    # for i in tqdm(range(n)):
    #     vi = thresholding(np.load(correct_paths[i]), r).flatten()
    #     for j in range(i+1, n):
    #         vj = thresholding(np.load(correct_paths[j]), r).flatten()
    #         thres_dist[i][j] = np.linalg.norm(vi-vj)

    # np.save('thresholding_0.95.npz', thres_dist)

    # clustering
    thres_dist = np.load('thresholding_0.95.npz.npy')

    i, j = np.triu_indices(thres_dist.shape[0], k=1)
    X = thres_dist[i, j]
    labels = [df_snp_correct.participant_id.iloc[i]+'\t\t'+df_snp_correct.diagnosis_x.iloc[i] for i in range(len(df_snp_correct))]
    Z = linkage(X, 'ward')
    fig = plt.figure(figsize=(50, 25))
    dn = dendrogram(Z, labels=labels, leaf_font_size=15)

    # visualize the results
    fig.suptitle('threasholding (0.95) dist', fontsize=16)
    fig.savefig('threasholding.pdf')



def denoisePath(path, noise_threshold=0.1):
    return join(dirname(path), basename(path).split('.npy')[0]+str(noise_threshold)+'.npy')


denoising_paths = [denoisePath(p) for p in correct_paths]



np.save('snp_paths.npy', correct_paths)


def denoising_dist_clustering():
    # calculate distance
    # n = len(df_snp_correct)
    # de_dist = np.zeros((n, n))
    # for i in tqdm(range(n)):
    #     vi = np.load(denoising_paths[i]).flatten()
    #     for j in range(i+1, n):
    #         vj = np.load(denoising_paths[j]).flatten()
    #         de_dist[i][j] = np.linalg.norm(vi-vj)

    # np.save('denoise.npy', de_dist)

    # clustering
    de_dist = np.load('denoise.npy')

    i, j = np.triu_indices(de_dist.shape[0], k=1)
    X = de_dist[i, j]
    labels = [df_snp_correct.participant_id.iloc[i]+'\t\t'+df_snp_correct.diagnosis_x.iloc[i] for i in range(len(df_snp_correct))]
    Z = linkage(X, 'ward')
    fig = plt.figure(figsize=(50, 25))
    dn = dendrogram(Z, labels=labels, leaf_font_size=15)

    # visualize the clustering results
    fig.suptitle('denoising (0.1) dist', fontsize=16)
    fig.savefig('denoising.pdf')


def df2rawpath(df, i):
    return join('/media/haohanwang/Elements/ADNI_CAPS','subjects', df.participant_id.iloc[i], df.session_id.iloc[i],
                'deeplearning_prepare_data', 'image_based', 't1_linear', 
                df.participant_id.iloc[i]+'_'+df.session_id.iloc[i]+'_'+'T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.pt')


raw_paths = [df2rawpath(df_snp, i) for i in range(len(df_snp))]

def rawData_clustering():

    n = len(df_snp)
    input_dist = np.zeros((n, n))
    for i in tqdm(range(n)):
        vi = torch.load(raw_paths[i]).squeeze().numpy().flatten()
        for j in range(i+1, n):
            vj = torch.load(raw_paths[j]).squeeze().numpy().flatten()
            input_dist[i][j] = np.linalg.norm(vi-vj)

    i, j = np.triu_indices(input_dist.shape[0], k=1)
    labels = [df_snp.participant_id[i]+'\t\t'+df_snp.diagnosis_x[i] for i in range(len(df_snp))]
    X = input_dist[i, j]
    Z = linkage(X, 'ward')
    fig = plt.figure(figsize=(50, 25))
    dn = dendrogram(Z, labels=labels, leaf_font_size=15)
    fig.suptitle('Raw Input dist', fontsize=16)
    fig.savefig('RawInput.pdf')

    nameWrite = open('raw_data_labels.txt', 'w')
    for name in dn['ivl']:
        nameWrite.writelines(name.split('\t')[0] + '\n')
    nameWrite.close()


if __name__ == '__main__':
    raw_dist_clustering()



