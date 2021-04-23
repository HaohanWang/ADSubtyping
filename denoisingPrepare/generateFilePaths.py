__author__ = 'Haohan Wang'



path_infos = [line.strip() for line in open('/media/haohanwang/Elements/ADNI_CAPS/split.stratified.0.csv')]

paths = []

for line in path_infos[1:]:
    items = line.split(',')
    p = items[-1] + '/' + items[0] + '/' + items[1] + '.npy'
    paths.append(p)

f = open('filePaths.txt', 'w')
for p in paths:
    f.writelines(p + '\n')
f.close()