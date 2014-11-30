#=========================================================================
# In this assignment it is necessary to implement Compression
#=========================================================================
import argparse
from scipy import misc, fftpack
from scipy.fftpack import dct
import cmath
import numpy as np
from matplotlib import pyplot
import scipy
from scipy.linalg import block_diag
from numpy import vstack
from scipy import signal

jpegstd = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                    [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99]
                    ])
zonal = np.array([
    [1, 1, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
])
zonal_best = np.array([
    [1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
])
thresholdmask = np.array([
    [1, 1, 0, 1, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
])
thresholdmask1 = np.array([
    [1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
])
haar_h0 = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
haar_h1 = np.array([1 / np.sqrt(2), -1 / np.sqrt(2)])
g0_daub = np.array([0.23037781, 0.71484657, 0.63088076, -
                    0.02798376, -0.18703481, 0.03084138, 0.03288301, -0.0105940])
g0_sym = np.array(
    [0.0322, -0.0126, -0.0992, 0.2979, 0.8037, 0.4976, -0.0296, -0.0758])
mats = {'jpeg': jpegstd, 'zonal': zonal, 'threshold': thresholdmask,
        'threshold1': thresholdmask1, 'zonalbest': zonal_best}


def chunks(l, n):
    if n < 1:
        n = 1
    return [l[i:i + n] for i in range(0, len(l), n)]


def computeWavelet(inputimage, waveletkernel, level):

    ret = np.copy(inputimage)
    h0, h1 = waveletkernel
    step = len(h0)
    m, _ = inputimage.shape
    w = m
    for _ in range(level):
        w /= 2
        for i in range(0, w):
            #             chunksup = np.array(chunks(ret[i, :2 * w], step))
            #             chunkslow = np.array(chunks(ret[w + i, :2 * w], step))
            #             print np.convolve(ret[i, :2 * w],h0,'same').shape
            ret[i, :w] = np.convolve(ret[i, : w], h0, 'same')
            ret[w + i, :w] = np.convolve(ret[w + i, : w], h1, 'same')
#             newrowup = []
#             for chunk in chunksup:
#                 	newrowup.append(np.inner(chunk, h0))
#             newrowlow = []
#             for chunk in chunkslow:
#                 newrowlow.append(np.inner(chunk, h1))
#             ret[i, :w] = np.array(newrowup).reshape(w)
#             ret[w + i, :w] = np.array(newrowlow).reshape(w)
        for i in range(0, w):
            ret[:w, i] = np.convolve(ret[:w, i], h0, 'same')
            ret[:w, w + i] = np.convolve(ret[:w, w + i], h0, 'same')
#             chunksup = np.array(chunks(ret[:2 * w, i], step))
#             chunkslow = np.array(chunks(ret[:2 * w, w + i], step))
#             newcolup = []
#             for chunk in chunksup:
#                 newcolup.append(np.inner(chunk, h0))
#             newrowlow = []
#             for chunk in chunkslow:
#                 newrowlow.append(np.inner(chunk, h1))
#             ret[:w, i] = np.array(newcolup).reshape(w)
#             ret[:w, w + i] = np.array(newrowlow).reshape(w)
    return ret
#     if wise == 'row':


def encodewavelet(inputimage, waveletkernel, level):
    ret = np.copy(inputimage)
    for _ in range(level):
        ret = np.dot(waveletkernel.T, np.dot(ret, waveletkernel))
    # ret = computeWavelet(ret, waveletkernel, level)
    return ret


def decodewavelet(inputimage, waveletkernel, level):
    ret = inputimage[:]
#     for _ in range(level):
#         ret = np.dot(waveletkernel, np.dot(ret, waveletkernel.T))
    return ret


def haarkernel(img):
    m, _ = img.shape

    return (np.vstack(((np.kron(np.eye(m / 2), haar_h0)), (np.kron(np.eye(m / 2), haar_h1)))).T)
    return (haar_h0, haar_h1)

'''
Returns the forward and the backward wavelet
'''


def daubechieskernel(inputimage):
    m, _ = inputimage.shape
    h0 = g0_daub[::-1]
    g1 = [h0[i] * -1. ** i for i in range(len(g0_daub))]
    h1 = g1[::-1]
    # return ((h0, h1), (g0_daub, g1))
    fwd = np.zeros((inputimage.shape))
    # We initizlize the matrix as having a stacked diagonal
    for i in range(0, (m / 2) - 2, 2):
        fwd[i, 2 * i:2 * i + 8] = h0
        fwd[(m / 2) + 1, 2 * i:2 * i + 8] = h1
    print fwd[254]
    bwd = None
    # fwd = np.vstack((np.kron(np.eye(m / 2), h0), np.kron(np.eye(m / 2), h1))).T
    # bwd = np.vstack(
    #     (np.kron(np.eye(m / 2), g0_daub), np.kron(np.eye(m / 2), g1))).T
    return (fwd, bwd)


def symletkernel(inputimage):
    m, _ = inputimage.shape
    h0 = g0_sym[::-1]
    g1 = [h0[i] * -1. for i in range(len(g0_daub))]
    h1 = g1[::-1]
    fwd = np.vstack((np.kron(np.eye(m / 2),)))

wavelets = {'haar': haarkernel, 'daub': daubechieskernel}


def main():
    args = parseArgs()
    if args.wavelet:
        kernel = wavelets[args.wavelet](args.inputimage)
#      if we return a tuple, we have two different functions for forward
#     transform and backward transform
        if args.wavelet == 'haar':
            fwd = kernel
            bwd = kernel
        else:
            fwd = kernel[0]
            bwd = kernel[1]
#         kernel = daubechieskernel(args.inputimage,None)
        encoded = encodewavelet(args.inputimage, fwd, 3)
# Removes the less important compnents, yet makes the picture quality
# worse
        encoded[encoded < args.threshold] = 0
#         Decode the thresholded image, will be in worse quality, but smaller
        decoded = decodewavelet(encoded, bwd, 3)
        difference = args.inputimage - decoded
        if args.o:
            misc.imsave(args.o + '_' + args.wavelet + '_encoded.tif', encoded)
            misc.imsave(
                args.o + '_' + args.wavelet + '_reconstructed.tif', decoded)
            misc.imsave(args.o + '_' + args.wavelet + '_diff.tif', difference)
    if args.quantmattype:
        encodedimgs = encode(args.inputimage, mats[args.quantmattype])
        decodedimg = decode(encodedimgs, args.inputimage.shape[0])
        if args.o:
            misc.imsave(args.o + '_reconstructed.tif', decodedimg)
        difference = args.inputimage - decodedimg
        if args.o:
            misc.imsave(args.o + '_diff.tif', difference)


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputimage', type=misc.imread)
    parser.add_argument('-quantmattype', choices=mats)
    parser.add_argument('-wavelet', choices=wavelets)
    parser.add_argument(
        '-t', '--threshold', help='Tht Threshold for the wavelet transforms, all values in the encoded wavelet will be truncated below threshold', type=int)
    parser.add_argument('-o', type=str, help='Outputfile')
    return parser.parse_args()


def decode(encodedimgs, rowlength):
    inverseimgs = inversetransform(encodedimgs)
    mergedimg = mergesubimages(inverseimgs, rowlength)
    return mergedimg
#     return readjustrange(mergedimg)


def mergesubimages(subimages, rowlength):
    ret = np.empty((rowlength, rowlength))
#     Actuially its 64 here
    for j in range(rowlength / (len(subimages) / rowlength)):
        for i in range(rowlength / (len(subimages) / rowlength)):
            subimg = subimages[
                (j * rowlength / (len(subimages) / rowlength)) + i]
            for q in range(len(subimg)):
                a = j * len(subimg) + q
                for p in range(len(subimg[0])):
                    b = i * len(subimg[0]) + p
                    ret[a, b] = subimg[q, p]
    return ret
#     return subimages.reshape((512,512))


def inversetransform(encodedimgs):
    ret = []
    for img in encodedimgs:
        ret.append(idct2(img))
    return np.array(ret)


def encode(rawimg, quantmat):
    #     adjustedimg = adjustrange(rawimg)
    rawimg = np.array(rawimg, dtype=float)
    subimages = np.array(list(constructsubimages(rawimg)))
    dcttransformedimgs = forwardtransform(subimages)
    quantimgs = quantitize(dcttransformedimgs, quantmat)
    return quantimgs


def dct2(data):
    return fftpack.dct(fftpack.dct(data.T, norm='ortho').T, norm='ortho')


def idct2(data):
    return fftpack.idct(fftpack.idct(data.T, norm='ortho').T, norm='ortho')


def constructsubimages(img, size=8):
    for i in range(0, len(img), size):
        for j in range(0, len(img[0]), size):
            yield img[i:i + size, j:j + size]


def forwardtransform(subimages):
    dctimages = []
    for image in subimages:
        dctimages.append(dct2(image))
    return np.array(dctimages)


def quantitize(dcttransformedimgs, quantmat):
    quantized = []
    for dcttransform in dcttransformedimgs:
        quantized.append(dcttransform * quantmat)
    return np.array(quantized)

if __name__ == '__main__':
    main()
