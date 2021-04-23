__author__ = 'Haohan Wang'

from clustering_visualize import drawFromPath
from tqdm import tqdm

session_ids = [line.strip() for line in open('raw_labels.txt')]

path_infos = [line.strip() for line in open('/media/haohanwang/Elements/ADNI_CAPS/split.stratified.0.csv')]

session_info = {}

for line in path_infos[1:]:
    items = line.split(',')
    session_info[items[0]] = (items[1], items[4], items[5])

session_paths = []
session_labels = []

for si in session_ids:
    session_paths.append('/media/haohanwang/Elements/saliency_map2/'+str(session_info[si][2])+'/' + si + '/' + session_info[si][0] + '.npy')
    session_labels.append(session_info[si][1])

for i in range(len(session_paths)):
    p = session_paths[i]
    l = session_labels[i]
    drawFromPath(p,l,'2d_view_clustered', i)