import pandas as pd
import numpy as np

df_mprage = pd.read_csv('../ADNI_meta/MPRAGEMETA.csv')
df_adni = pd.read_csv('../ADNI_meta/ADNIMERGE.csv')
df_mprage_annual_2 = pd.read_csv('../ADNI_meta/ADNI_CompleteAnnual2YearVisitList_8_22_12.csv')

filelist = glob('../Alzheimer/ADNI1_Annual_2_Yr_1.5T/ADNI/*/*/*/*/*')


prefer = [
'MPR-R; GradWarp; B1 Correction; N3; Scaled_2',
'MPR-R; GradWarp; B1 Correction; N3; Scaled',
'MPR; GradWarp; B1 Correction; N3; Scaled_2',
'MPR; GradWarp; B1 Correction; N3; Scaled',
'MPR-R; GradWarp; N3; Scaled_2',
'MPR-R; GradWarp; N3; Scaled',
'MPR; GradWarp; N3; Scaled_2',
'MPR; GradWarp; N3; Scaled',
'MPR-R; ; N3; Scaled_2',
'MPR-R; ; N3; Scaled',
'MPR; ; N3; Scaled_2',
'MPR; ; N3; Scaled'
]

visit = ['Screening', 'Month 12', 'Month 24']
sub_list = df_mprage_annual_2['PTID'].drop_duplicates().values
sub_select = []
for sub in sub_list:
    for v in visit:
        entry = df_mprage_annual_2[(df_mprage_annual_2['PTID'] == sub) & \
                                   (df_mprage_annual_2['Visit'] == v)]
        for p in prefer:
            e = entry[entry['Sequence'].str.contains(p)]
            if e.shape[0] != 0:
                sub_select.append(e)
                break
        if e.shape[0] == 0:
            print('Error: no match!')
            
df_annual2 = pd.concat(objs=sub_select,axis=0,join="outer")     

viscode = ['bl', 'm12', 'm24']
dx = []
for i in range(len(df_annual2)):
    vis =  viscode[visit.index(df_annual2.iloc[i]['Visit'])]
    dxi = df_adni[(df_adni['VISCODE'] == vis) & (df_adni['PTID'] == df_annual2.iloc[i]['PTID'])]['DX'].values[0]
    if type(dxi) == float and np.isnan(dxi):
        dxi = df_adni[(df_adni['VISCODE'] == vis) & (df_adni['PTID'] == df_annual2.iloc[i]['PTID'])]['DX_bl'].values[0]
    dx.append(dxi)

cnt = 0
for i in range(len(dx)):
    ori_diag = df_annual2.iloc[i]['Screen.Diagnosis']
    if dx[i] == 'Dementia':
        if ori_diag != 'AD':
            cnt += 1
            print(ori_diag + ' -> AD')
        dx[i] = 'AD'
    elif dx[i] == 'CN':
        if ori_diag != 'NL':
            cnt += 1
            print(ori_diag + ' -> NL')
        dx[i] = 'NL'
    elif dx[i] == 'MCI':
        if ori_diag != 'MCI':
            cnt += 1
            print(ori_diag + ' -> MCI')
        dx[i] = 'MCI'
    else:
        print(ori_diag + '-> ' + str(dx[i]))
        cnt += 1
        dx[i] = dx[i]

df_annual2['DX'] = dx

df_annual2.to_csv('adni1_anual2_1.5T.tsv',index=None, sep='\t')