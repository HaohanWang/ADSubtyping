import numpy as np
import ipyvolume as ipv

from glob import glob
from tqdm import tqdm

from os.path import join, dirname, basename, exists
from multiprocessing.pool import Pool

def save_ipv(path_list, pid):
    
    tbar = tqdm(enumerate(path_list), total=len(path_list), postfix=str(pid))
    for i, path in tbar:
        data = np.load(path)
        sess = basename(path).split('.')[0]
        out = sess + '.html'
        outpath = join(dirname(path), out)
        if exists(outpath):
            continue
        try:
            ipv.pylab.clear()
            ipv.volshow(data, level=[0.2,0.2,1])
            ipv.pylab.save(outpath)
        except:
            pass
    

def main():
    n_procs = 7
    files = glob('saliency_map/*/sub-ADNI*/ses*.npy')
    files_splits = np.array_split(files, n_procs)
    assert sum([len(v) for v in files_splits]) == len(files)
    p = Pool(n_procs)
    for i, split in enumerate(files_splits):
        p.apply_async(save_ipv, args=(split, i))
    print("Waiting for all subprocesses done...")
    p.close()
    p.join()
    print("All subprocesses done.")

if __name__ == "__main__":
    main()    




