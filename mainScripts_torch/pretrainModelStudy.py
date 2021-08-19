__author__ = 'Haohan Wang'

import numpy as np

import torch

m = torch.load('../pretrainModels/best_model/fold_0/model_best.pth.tar')

model = m['model']

for (k, v) in model.items():
    print (k)
    weights = v.cpu().numpy()
    print (weights.shape)
    np.save('../pretrainModels/best_model/fold_0/npy_weights/' + k + '.npy', weights)