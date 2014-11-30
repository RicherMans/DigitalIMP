'''
Created on Nov 29, 2014

@author: richman
'''
import argparse
from scipy import misc
from numpy import dot, linalg, sqrt
import pylab
import os
import numpy as np


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputdir', type=str, help='directory of the pictures')
    return parser.parse_args()


def pca1(arr):
    px, py, veclength = arr.shape
#     TODO: No idea why it doesnt work!
#     Calculate mean for every pixel
    ret = np.empty((arr.shape), dtype=float)
    rowmean = np.matrix(np.mean(arr, axis=0))
    covariances = []
    for i in range(len(arr)):
        sumcov = 0
        for j in range(len(arr[0])):
            sumcov += dot((np.matrix(arr[i, j]) - rowmean[i]).T,
                          (np.matrix(arr[i, j]) - rowmean[i]))
        covariances.append(sumcov)
    for cov in covariances:
        w, v = np.linalg.eigh(cov)
        print max(w)
#     cov = sum(sum(ret))
#     print cov


def main():
    args = parseArgs()
    picpaths = [os.path.join(os.getcwd(), args.inputdir, x)
                for x in os.listdir(args.inputdir)]
    inputimages = [misc.imread(pic) for pic in picpaths]
#     First of all we transform the input from (6,N,N) to (N,N,6), getting
#   for every pixel n,n a 6 dimensional vector.
    imgs = np.array(inputimages).T

    pca1(imgs)
#     pca1(stackedimgs)
    # show the images
#     pylab.figure()
#     pylab.gray()
#     pylab.imshow(proj)
#
#     pylab.show()

if __name__ == '__main__':
    main()
