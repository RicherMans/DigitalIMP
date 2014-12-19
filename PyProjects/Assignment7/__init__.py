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
h0_cohen = np.array(
                    [0, 0.0019, -0.0019, -0.017, 0.0119, 0.0497, -0.0773, -0.0941, 0.4208, 0.8259, 0.4208, -0.0941, -0.0773, 0.0497, 0.0119, -0.017, -0.0019, 0.0010]
                    )
h1_cohen = np.array(
                    [0, 0, 0, 0.0144, -0.0145, -0.0787, 0.0404, 0.4178, -0.7589, 0.4178, 0.0404, -0.0787, -0.0145, 0.0144, 0, 0, 0, 0]
                    )
mats = {'jpeg': jpegstd, 'zonal': zonal, 'threshold': thresholdmask,
        'threshold1': thresholdmask1, 'zonalbest': zonal_best}


def chunks(l, n):
    if n < 1:
        n = 1
    return [l[i:i + n] for i in range(0, len(l), n)]


def encodewavelet(inputimage, waveletkernel, level):
    ret = np.copy(inputimage)
    for _ in range(level):
        ret = np.dot(waveletkernel.T, np.dot(ret, waveletkernel))
    # ret = computeWavelet(ret, waveletkernel, level)
    return ret


def decodewavelet(inputimage, waveletkernel, level):
    ret = inputimage[:]
    for _ in range(level):
        ret = np.dot(waveletkernel, np.dot(ret, waveletkernel.T))
    return ret


def haarkernel(img):
    m, _ = img.shape
    return (np.vstack(((np.kron(np.eye(m / 2), haar_h0)), (np.kron(np.eye(m / 2), haar_h1)))).T)


def daubechieskernel(inputimage):
    m, _ = inputimage.shape
    h0 = g0_daub[::-1]
    g1 = [h0[i] * np.power(-1., i) for i in range(len(g0_daub))]
    h1 = g1[::-1]
    fwd = np.zeros((inputimage.shape))
    # We initizlize the matrix as having a stacked diagonal
    # Using two different submatrices, where the lower and the upper part are
    # (h0, h1) respectively
    for i in range(0, m / 2):
        for j in range(len(h0)):
            fwd[i, (2 * i + j) % m] = h0[j]
            fwd[(m / 2) + i, (2 * i + j) % m] = h1[j]
    return fwd.T

def cohenkernel(inputimage):
    m, _ = inputimage.shape
    fwd = np.zeros((inputimage.shape))
    for i in range(0, m / 2):
        for j in range(len(h0_cohen)):
            fwd[i, (2 * i + j) % m] = h0_cohen[j]
            fwd[(m / 2) + i, (2 * i + j) % m] = h1_cohen[j]
    return fwd.T
def symletkernel(inputimage):
    m, _ = inputimage.shape
    h0 = g0_sym[::-1]
    g1 = [h0[i] * np.power(-1., i) for i in range(len(g0_daub))]
    h1 = g1[::-1]

    fwd = np.zeros((inputimage.shape))
    for i in range(0, m / 2):
        for j in range(len(h0)):
            fwd[i, (2 * i + j) % m] = h0[j]
            fwd[(m / 2) + i, (2 * i + j) % m] = h1[j]
    return fwd.T

wavelets = {'haar': haarkernel, 'daub': daubechieskernel,
            'sym': symletkernel , 'cohen':cohenkernel}


def main():
    args = parseArgs()
    if args.wavelet:
        kernel = wavelets[args.wavelet](args.inputimage)
        encoded = encodewavelet(args.inputimage, kernel, 3)
# Removes the less important compnents, yet makes the picture quality
# worse
        encoded[encoded < args.threshold] = 0
#         Decode the thresholded image, will be in worse quality, but smaller
        decoded = decodewavelet(encoded, kernel, 3)
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
            yield img[i: i + size, j: j + size]


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
