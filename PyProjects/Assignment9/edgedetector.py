'''
Created on Nov 28, 2014

@author: richman
'''

import argparse
from scipy import misc
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.signal.signaltools import convolve2d
import scipy.ndimage as ndi
from matplotlib import pyplot

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputimage', type=misc.imread)
    return parser.parse_args()

def non_maximal_edge_suppresion(mag, orient):
    """Non Maximal suppression of gradient magnitude and orientation."""
    # bin orientations into 4 discrete directions
    abin = ((orient + np.pi) * 4 / np.pi + 0.5).astype('int') % 4

    mask = np.zeros(mag.shape, dtype='bool')
    mask[1:-1,1:-1] = True
    edge_map = np.zeros(mag.shape, dtype='bool')
    offsets = ((1,0), (1,1), (0,1), (-1,1))
    for a, (di, dj) in zip(range(4), offsets):
        cand_idx = np.nonzero(np.logical_and(abin==a, mask))
        for i,j in zip(*cand_idx):
            if mag[i,j] > mag[i+di,j+dj] and mag[i,j] > mag[i-di,j-dj]:
                edge_map[i,j] = True
    return edge_map

def main():
    args = parseArgs()
#     maarhimg = filter_maar_hidlret(args.inputimage)
#     misc.imsave('marr_hidlret.tif', maarhimg)
#     misc.imsave('canny.tif',cannyimg.grad)
#     sobelfilterd = filter_w_mask(args.inputimage, 'sobel')
#     misc.imsave('sobel.tif',sobelfilterd)
#      
#     prewittfiltered = filter_w_mask(args.inputimage, 'prewitt')
#     misc.imsave('prewitt.tif', prewittfiltered)

#     robertsfiltered = filter_w_mask(args.inputimage, 'roberts')
#     misc.imsave('roberts.tif', robertsfiltered)
    
    cannyedge = filter_canny(args.inputimage, 'prewitt')
    misc.imsave('canny_edge.tif',cannyedge)

def filter_w_mask(img,filttype='sobel'):
    gx,gy = calculategradient(img, filttype)
    g = gradientmagnitude(gx, gy)
    g[g > 255] = 255
    return g

def filter_canny(inputimage, filttype):

    gaussianfiltered = gaussian_filter(inputimage, 2.,mode='constant')
    gx, gy = calculategradient(gaussianfiltered, filttype)
    mag = gradientmagnitude(gx, gy)
    angle = (directionangle(gx, gy)*180/np.pi)+180
#     maximum suppression
    width, height = angle.shape
#     Change the angles to either being -45,0,45,90
      
    angle[(angle < 22.5) | (angle > 337.5) & (angle < 202.5) | (angle > 157.5) ] = 0.
    angle[((angle >= 22.5) & (angle <= 67.5)) | ((angle >= 202.5) & (angle <= 247.5))] = 45.
    angle[((angle > 67.5) & (angle < 112.5)) | ((angle >= 247.5) & (angle <= 292.5))] = 90.
    angle[((angle <= 157.5) & (angle > 112.5)) | ((angle >= 292.5) & (angle <= 337.5))] = -45.
      
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
    th = m*0.1
    tl = th/2
    gnh = np.zeros((width, height),dtype=float)
    gnl = np.zeros((width, height),dtype=float)
    for x in range(1,width-1):
        for y in range(1,height-1):
            if mag_sup[x][y] >= th:
                gnh[x][ y] = mag_sup[x][y]
            if mag_sup[x][y] >= tl:
                gnl[x][ y] = mag_sup[x][y]
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
                gnh[i, j] = 1
                traverse(i, j)
    gnh[gnh>255]=255
    return gnh


def filter_maar_hidlret(inputimage):
    filteredimg = filterimg(inputimage, maar_hidlret_kernel((25, 25), 4))
    laplacianimg = applylaplacian(filteredimg)
    misc.imsave('test.tif', laplacianimg)
    return zero_crossings(laplacianimg)


def sgn(n):
    if n > 0:
        return 1
    else:
        return -1


def gradientmagnitude(gx, gy):
    return np.hypot(gx,gy)


def directionangle(gx, gy):
    return np.arctan2(gy , gx)
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
    threshold = 10
    offset = 1
    ret = np.zeros((arr.shape),dtype=bool)
    for i in range(offset, len(arr) - offset):
        for j in range(offset, len(arr[0]) - offset):
            if sgn(arr[i + offset][j]) != sgn(arr[i - offset][j]) and (abs(arr[i + offset][j]) - abs(arr[i - offset][j])) > threshold:
                ret[i, j] = 1
                continue
            if sgn(arr[i][j + offset]) != sgn(arr[i][j - offset]) and (abs(arr[i][j + offset]) - abs(arr[i][j - offset])) > threshold:
                ret[i, j] = 1
                continue
            if sgn(arr[i + offset][j - offset]) != sgn(arr[i - offset][j + offset]) and (abs(arr[i + offset][j - offset]) - abs(arr[i - offset][j + offset])) > threshold:
                ret[i, j] = 1
                continue
            if sgn(arr[i + offset][j + offset]) != sgn(arr[i - offset][j - offset]) and (abs(arr[i + offset][j + offset]) - abs(arr[i - offset][j - offset])) > threshold:
                ret[i, j] = 1
                continue
    return ret


def calculategradient(arr, filttype='sobel'):
    ''' Three different types, prewitt,sobel and roberts'''
    def sobel():
        return (np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))

    def prewitt():
        return (np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]), np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))

    def roberts():
        return (np.array([[1, 0], [0, -1]]), np.array([[0, 1], [-1, 0]]))
    masks = {'sobel': sobel(), 'prewitt': prewitt(), 'roberts': roberts()}
#     masks = {'sobel': ndi.filters.sobel, 'prewitt': ndi.filters.prewitt, 'roberts': None}
    x, y = masks[filttype]
#     Once again remove the boundaries
    gx = filterimg(arr,x)
    gy = filterimg(arr,y)
    return gx, gy


def applylaplacian(arr):
    mask = np.array([[1, 1, 1], [1, -9, 1], [1, 1, 1]])
    return filterimg(arr, mask)


def filterimg(arr, mask):
    return convolve2d(arr,mask)


def maar_hidlret_kernel(size, sigma):
    m, n = [(ss - 1) / 2. for ss in size]
    x, y = np.ogrid[-m:m + 1, -n:n + 1]
    return np.exp(-1 * (x ** 2 + y ** 2) / 2 * sigma ** 2)

if __name__ == '__main__':
    main()
