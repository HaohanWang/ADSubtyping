import numpy as np
from tqdm import tqdm

from os.path import join, dirname, basename, exists
from multiprocessing.pool import Pool

import pyfftw

def Denoising(oriPath, noise_threshold, outpath):
    oriData = np.load(oriPath)
    fftf=pyfftw.interfaces.numpy_fft.fftn(oriData)
    amplitudeList = []

    for dim1 in range(fftf.shape[0]):
        for dim2 in range(fftf.shape[1]):
            for dim3 in range(fftf.shape[-1]):
                amplitudeList.append(pow(abs(fftf[dim1][dim2][dim3]),2))

    amplitudeList.sort()


    threshold = int(len(amplitudeList)*noise_threshold)

    for dim1 in range(fftf.shape[0]):
        for dim2 in range(fftf.shape[1]):
            for dim3 in range(fftf.shape[-1]):
                if pow(abs(fftf[dim1][dim2][dim3]),2)< amplitudeList[threshold]:
                    fftf[dim1][dim2][dim3] = 0
                else:
                    tmp0 = fftf[dim1][dim2][dim3]
                    tmp2 = pow(abs(tmp0),2)
                    fftf[dim1][dim2][dim3] = np.sqrt((tmp2 - amplitudeList[threshold]) / tmp2) * tmp0

    nfft=pyfftw.interfaces.numpy_fft.ifftn(fftf)
    np.save(outpath, np.real(nfft))
    

def denoise(path_list, noise_threshold, pid):
    
    tbar = tqdm(enumerate(path_list), total=len(path_list), postfix=str(pid))
    for i, path in tbar:
        outpath = join(dirname(path), basename(path).split('.npy')[0]+str(noise_threshold)+'.npy')
       # if exists(outpath):
       #     continue
        try:
            Denoising(path, noise_threshold, outpath)
        except:
            pass
    

def main():
    n_procs = 7
    noise_threshold = 0.1
    files = np.load('snp_paths.npy')
    files = [str(s) for s in files]
    files_splits = np.array_split(files, n_procs)
    assert sum([len(v) for v in files_splits]) == len(files)
    p = Pool(n_procs)
    for i, split in enumerate(files_splits):
        p.apply_async(denoise, args=(split, noise_threshold, i))
    print("Waiting for all subprocesses done...")
    p.close()
    p.join()
    print("All subprocesses done.")

if __name__ == "__main__":
    main()    




