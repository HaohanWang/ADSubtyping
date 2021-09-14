__author__ = 'Haohan Wang'

from scipy.ndimage import rotate, interpolation
import numpy as np


class MRIDataAugmentation():
    def __init__(self, imgShape, augProb):
        self.height = imgShape[0]
        self.width = imgShape[1]
        self.depth = imgShape[2]
        self.augProb = augProb

        self.funcs_pool = [self.rotate_img, self.scale_img, self.translate_img]

    def augmentData_batch(self, imgs):
        for i in range(imgs.shape[0]):
            imgs[i,:,:,:,0] = self.augmentData_single(imgs[i,:,:,:,0])
        return imgs

    def augmentData_single(self, img):
        if np.random.random() < self.augProb:
            numFunc = np.random.randint(1, 4)
            funcs = np.random.choice(self.funcs_pool, numFunc, replace=False)
            np.random.shuffle(funcs)
            for func in funcs:
                img = func(img)
        return img


    def rotate_img(self, image):
        theta = np.random.random()*20-10.0
        return rotate(image, float(theta), reshape=False, order=0, mode='nearest')

    def scale_img(self, image):
        order = 0
        factor = np.random.random()*0.2 + 0.9

        zheight = int(np.round(factor * self.height))
        zwidth = int(np.round(factor * self.width))
        zdepth = self.depth

        if factor < 1.0:
            newimg = np.zeros_like(image)
            row = (self.height - zheight) // 2
            col = (self.width - zwidth) // 2
            layer = (self.depth - zdepth) // 2
            newimg[row:row + zheight, col:col + zwidth, layer:layer + zdepth] = interpolation.zoom(image, (
            float(factor), float(factor), 1.0), order=order, mode='nearest')[0:zheight, 0:zwidth, 0:zdepth]

            return newimg

        elif factor > 1.0:
            row = (zheight - self.height) // 2
            col = (zwidth - self.width) // 2
            layer = (zdepth - self.depth) // 2

            newimg = interpolation.zoom(image[row:row + zheight, col:col + zwidth, layer:layer + zdepth],
                                        (float(factor), float(factor), 1.0), order=order, mode='nearest')

            extrah = (newimg.shape[0] - self.height) // 2
            extraw = (newimg.shape[1] - self.width) // 2
            extrad = (newimg.shape[2] - self.depth) // 2
            newimg = newimg[extrah:extrah + self.height, extraw:extraw + self.width, extrad:extrad + self.depth]

            return newimg

        else:
            return image

    def translate_img(self, image):
        offset  = list(np.random.randint(-5,5, size=3))
        return interpolation.shift(image, offset, order=0, mode='nearest')