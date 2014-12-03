import argparse
from scipy import misc
from scipy.ndimage.filters import convolve as convolveim
from scipy.ndimage import gaussian_filter
import numpy as np
import sys
from numpy import real


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputimage', type=misc.imread)
    parser.add_argument(
        '-a', help='the parameter a in the equation, which is equal to b', default=0.1, type=float)
    parser.add_argument('-T', help='the parameter T', default=1, type=float)
#     parser.add_argument('-o',help='outputs the produced image')
    return parser.parse_args()


def main():
    args = parseArgs()
    blurredkernel = blurringkernel(
        args.inputimage.shape, args.T, args.a, args.a)
    motionblurredimg = applymotionblur(
        args.inputimage, blurredkernel, blurringfilter)
    misc.imsave('blurred.jpg', motionblurredimg)
    blur_noise_img = motionblurredimg + gaussiannoise(0, 650, motionblurredimg)
    misc.imsave('blurred_w_gauss.jpg', blur_noise_img)
    restored_inverse = filterimg(
        blur_noise_img, blurredkernel, inversefilter)
    misc.imsave('restored_inversefilt.jpg', restored_inverse)
    restored_wiener = filterimg(
        blur_noise_img, blurredkernel, wienerfilter)
    misc.imsave('restored_wiener_param1.jpg', restored_wiener)
    restored_wiener_2 = filterimg(
        blur_noise_img, blurredkernel, wienerfilter, 2.)
    misc.imsave('restored_wiener_param2.jpg', restored_wiener_2)
    restored_wiener_10 = filterimg(
        blur_noise_img, blurredkernel, wienerfilter, 10.)
    misc.imsave('restored_wiener_param10.jpg', restored_wiener_10)
    restored_wiener_011 = filterimg(
        blur_noise_img, blurredkernel, wienerfilter, .01)
    misc.imsave('restored_wiener_param011.jpg', restored_wiener_011)
    restored_wiener_001 = filterimg(
        blur_noise_img, blurredkernel, wienerfilter, .001)
    misc.imsave('restored_wiener_param001.jpg', restored_wiener_001)
    restored_wiener_050 = filterimg(
        blur_noise_img, blurredkernel, wienerfilter, .05)
    misc.imsave('restored_wiener_param05.jpg', restored_wiener_050)


def gaussiannoise(mean, var, data):
    if var == 0:
        return np.zeros(data.shape)
    else:
        return np.random.normal(mean, np.sqrt(var), data.shape)


def eucl_dist(i, j, M, N):
    return np.sqrt((i - (M / 2)) ** 2 + (j - (N / 2)) ** 2)


def normalize(arr):
    for i in range(len(arr)):
        normalizer = max(arr[i])
        for j in range(len(arr[0])):
            arr[i][j] = arr[i][j] / normalizer * 255


def blurringkernel(shape, T, a, b):
    xx, yy = shape
    x, y = np.ogrid[(-xx / 2):(xx / 2), (-yy / 2):yy / 2]
    q = (np.pi * (x * a + y * b))
    q[np.where(q == 0)] = T
    return (T / q) * np.sin(q) * np.exp(-1j * q)


def blurringfilter(dftarr, kernel):
    return applykernel(dftarr, kernel)


def powerspectrum(img):
    ret = img * img
    return ret


def wienerkernel(origimg, blurredkernel, param, k):
    '''
    Calculates the wiener kernel, which is H*/(H^2+param*(S_n/S_f))
    '''
    h = blurredkernel
    return np.conj(h).T / ((abs(h) ** 2) + (param * k))


def wienerfilter(dftarr, blurredkernel, param=1, k=1):
    kernel = wienerkernel(dftarr,  blurredkernel, param, k)
    return applykernel(dftarr, kernel)

'''
Inverse filter is calculated using F^ = G(u,v) H(u,v), since we know H(u,v) , which is the blurring function
we can simply return the inverse of H(u,v)
'''


def inversekernel(kernel):
    return 1. / kernel


def inversefilter(dftarr, kernel):
    return applykernel(dftarr, inversekernel(kernel))


def applykernel(arr, kernel):
    return arr * kernel


def extractReal(img):
    return np.array([[(img[j][i]).real for i in range(len(img))] for j in range(len(img[0]))])


def center(imagearr):
    m, n = imagearr.shape
    x, y = np.ogrid[0:m, 0:n]
    return imagearr * ((-1) ** (x + y))


def applymotionblur(imagearray, kernel, filt, *args):
    imagearray = center(imagearray)
    dftrans = np.fft.fft2(imagearray)
    filteredimg = filt(dftrans, kernel, *args)
    inversefft = np.fft.ifft2(filteredimg)
    realimg = extractReal(inversefft)
    return center(realimg)


def filterimg(blurredimg, blurkernel, filt, *args):
    imagearray = center(blurredimg)
    dfttrans = np.fft.fft2(imagearray)
    filterredimg = filt(dfttrans,  blurkernel, *args)
    inversefft = np.fft.ifft2(filterredimg)
    realimg = extractReal(inversefft)
    return center(realimg)


if __name__ == '__main__':
    main()
