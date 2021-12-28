import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import StratifiedKFold

import os


data_dir = '.'
df = pd.read_csv('ADNI1_Annual_2_Yr_1.5T.tsv', sep='\t')
df_bl = df[df['session_id'] == 'ses-M00']
sub_list = df_bl.participant_id.values
sub_diag = df_bl.diagnosis.values

# X = np.arange(len(sub_list))
# y = sub_list
kf = StratifiedKFold(n_splits=11, random_state=1234)
folds = []

for train_index, test_index in kf.split(sub_list, sub_diag):
    folds.append(test_index)

for a, b in combinations(folds, 2):
    assert len(set(a) & set(b)) == 0

num_fold = 5
for fold_idx in range(num_fold):
    records = {}
    for i, indices in enumerate(folds):
        if i == (fold_idx * 2) or i == (fold_idx * 2) + 1:
            for j in indices:
                records[sub_list[j]]='val'
        elif i == 10:
            for j in indices:
                records[sub_list[j]]='test_val'
        else:
            for j in indices:
                records[sub_list[j]]='train'
    df_idx = df.copy()
    split_idx = []
    for i in range(df_idx.shape[0]):
        split_idx.append(records[df_idx['participant_id'][i]])
    df_idx['split'] = split_idx
    output_filename = os.path.join(data_dir, 'split.stratified.{}.csv'.format(fold_idx))
    df_idx.to_csv(output_filename, index=False)