import pyfftw
import numpy as np
import time

import ipyvolume as ipv
from utility.loadResults import loadNPYFile

from scipy.ndimage import zoom

filePath = '/media/haohanwang/Elements/saliency_map/'


#Here we can change the noise threshold: eg. noise_threshold = 0.1 or 0.2 or, ..., 0.5
def Denoising(oriData, subject, noise_threshold):
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
    # np.save(filePath + 'test/' + subject + '/'+'denoise_rst.npy',nfft)
    return np.real(nfft)

def reshapeData(data):

    result = zoom(data, [0.1, 0.1, 0.1])
    return result

if __name__ == '__main__':
    startTime = time.time()

    subj = 'sub-ADNI002S0559'
    sess = 'ses-M00'
    trainFlag = True
    noise_threshold = 0.95

    data = loadNPYFile(subj, sess, filePath)

    print data.shape
    print np.max(data)
    print np.std(data)

    result = Denoising(data, sess, noise_threshold=noise_threshold)

    # result = reshapeData(data)
    # result = reshapeData(result)

    # q = np.quantile(data.reshape([169*208*179]), 0.9)
    # result = np.copy(data)
    # result[result<q] = 0


    print np.max(result)
    print np.std(result)

    # print result

    ipv.pylab.clear()
    ipv.volshow(result, level=[0.2,0.2,1.0])
    if trainFlag:
        ipv.pylab.save(filePath + 'train/' + subj + '/' + sess +'_denoise_rst.html')
    else:
        ipv.pylab.save(filePath + 'test/' + subj + '/'+ sess + '_denoise_rst.html')

    endTime = time.time()

    print 'time elapsed:', endTime - startTime
