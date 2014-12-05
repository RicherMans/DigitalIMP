'''
Created on Nov 28, 2014

@author: richman
'''

import argparse
from scipy import misc
import numpy as np
from matplotlib import pyplot
from scipy import signal
from scipy.ndimage.filters import gaussian_filter


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputimage', type=misc.imread)
    return parser.parse_args()


def main():
    args = parseArgs()
    maarhimg = filter_maar_hidlret(args.inputimage)
    misc.imsave('marr_hidlret.tif', maarhimg)
    sobelfiltered = filter_canny(args.inputimage, 'sobel')
    misc.imsave('canny_sobel.tif', sobelfiltered)

    prewittfiltered = filter_canny(args.inputimage, 'prewitt')
    misc.imsave('canny_prewitt.tif', prewittfiltered)

    robertsfiltered = filter_canny(args.inputimage, 'roberts')
    misc.imsave('canny_roberts.tif', robertsfiltered)


def filter_canny(inputimage, filttype):
    gaussianfiltered = gaussian_filter(inputimage, 4)
    gx, gy = calculategradient(gaussianfiltered, filttype)
    mag = gradientmagnitude(gx, gy)
    angle = directionangle(gx, gy) / np.pi * 180
#     maximum suppression
    width, height = inputimage.shape
    print angle[0]
    mag_sup = np.copy(mag)
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            if angle[x][y] == 0:
                if (mag[x][y] <= mag[x][y + 1]) or \
                   (mag[x][y] <= mag[x][y - 1]):
                    mag_sup[x][y] = 0
            elif angle[x][y] == 45:
                if (mag[x][y] <= mag[x - 1][y + 1]) or \
                   (mag[x][y] <= mag[x + 1][y - 1]):
                    mag_sup[x][y] = 0
            elif angle[x][y] == 90:
                if (mag[x][y] <= mag[x + 1][y]) or \
                   (mag[x][y] <= mag[x - 1][y]):
                    mag_sup[x][y] = 0
            else:
                if (mag[x][y] <= mag[x + 1][y + 1]) or \
                   (mag[x][y] <= mag[x - 1][y - 1]):
                    mag_sup[x][y] = 0
#     Edge linking
    m = np.max(mag_sup)
    th = 0.2 * m
    tl = 0.1 * m
    print m
    gnh = np.zeros((width, height))
    gnl = np.zeros((width, height))

    for x in range(width):
        for y in range(height):
            if mag_sup[x][y] >= th:
                gnh[x][y] = mag_sup[x][y]
            if mag_sup[x][y] >= tl:
                gnl[x][y] = mag_sup[x][y]
    gnl = gnl - gnh

    def traverse(i, j):
        x = [-1, 0, 1, -1, 1, -1, 0, 1]
        y = [-1, -1, -1, 0, 0, 1, 1, 1]
        for k in range(8):
            if gnh[i + x[k]][j + y[k]] == 0 and gnl[i + x[k]][j + y[k]] != 0:
                gnh[i + x[k]][j + y[k]] = 1
                traverse(i + x[k], j + y[k])

    for i in range(1, width - 1):
        for j in range(1, height - 1):
            if gnh[i][j]:
                gnh[i][j] = 1
                traverse(i, j)
    gnh[gnh > 255.] = 255.
    return gnh


def filter_maar_hidlret(inputimage):
    filteredimg = filterimg(inputimage, maar_hidlret_kernel((25, 25), 4))
    laplacianimg = applylaplacian(filteredimg)
    return zero_crossings(laplacianimg)


def sgn(n):
    if n > 0:
        return 1
    else:
        return -1


def gradientmagnitude(gx, gy):
    return np.sqrt(gx ** 2 + gy ** 2)


def directionangle(gx, gy):
    # lim of arctan(x) is pi/2
    gx[np.where(gx == 0)] = np.pi / 2
    gy[np.where(gy == 0)] = np.pi / 2
    return np.arctan(gx / gy)
# Better use the library, will be otherwise hard to read


def gauss2dkernel(shape, sigma):
    xx, yy = shape
#     m, n = [(ss -1) / 2. for ss in shape]
    x, y = np.mgrid[0:xx, 0:yy]
    d = eucl_dist(x, y, len(x), len(y))
    h = np.exp(-(d ** 2 / (2. * sigma ** 2)))
#     h[ h < np.finfo(h.dtype).eps * h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def eucl_dist(i, j, M, N):
    return np.sqrt((i - (M / 2)) ** 2 + (j - (N / 2)) ** 2)


def zero_crossings(arr):
    threshold = 5
    offset = 1
    ret = np.zeros((arr.shape))
    for i in range(offset, len(arr) - offset):
        for j in range(offset, len(arr[0]) - offset):
            if sgn(arr[i + offset][j]) != sgn(arr[i - offset][j]) and abs(arr[i + offset][j]) - abs(arr[i - offset][j]) > threshold:
                ret[i, j] = 255
                continue
            if sgn(arr[i][j + offset]) != sgn(arr[i][j - offset]) and abs(arr[i][j + offset]) - abs(arr[i][j - offset]) > threshold:
                ret[i, j] = 255
                continue
            if sgn(arr[i + offset][j - offset]) != sgn(arr[i - offset][j + offset]) and abs(arr[i + offset][j - offset]) - abs(arr[i - offset][j + offset]) > threshold:
                ret[i, j] = 255
                continue
            if sgn(arr[i + offset][j + offset]) != sgn(arr[i - offset][j - offset]) and abs(arr[i + offset][j + offset]) - abs(arr[i - offset][j - offset]) > threshold:
                ret[i, j] = 255
                continue
    return ret


def calculategradient(arr, type='sobel'):
    ''' Three different types, prewitt,sobel and roberts'''
    def sobel():
        return (np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))

    def prewitt():
        return (np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]), np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))

    def roberts():
        return (np.array([[1, 0], [0, -1]]), np.array([[0, 1], [-1, 0]]))
    masks = {'sobel': sobel(), 'prewitt': prewitt(), 'roberts': roberts()}
    x, y = masks[type]
    gx = signal.convolve2d(arr, x)
    gy = signal.convolve2d(arr, y)
    return gx, gy


def applylaplacian(arr):
    mask = np.array([[1, 1, 1], [1, -9, 1], [1, 1, 1]])
    return signal.convolve2d(arr, mask)


def filterimg(arr, mask):
    return signal.convolve2d(arr, mask)


def maar_hidlret_kernel(size, sigma):
    m, n = [(ss - 1) / 2. for ss in size]
    x, y = np.ogrid[-m:m + 1, -n:n + 1]
    return np.exp(-1 * (x ** 2 + y ** 2) / 2 * sigma ** 2)

if __name__ == '__main__':
    main()
