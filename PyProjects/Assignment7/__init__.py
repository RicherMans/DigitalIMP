#===============================================================================
# In this assignment it is necessary to implement Compression
#===============================================================================
import argparse
from scipy import misc, fftpack
from scipy.fftpack import dct
import cmath
import numpy as np
from matplotlib import pyplot
import scipy
from scipy.linalg import block_diag
from numpy import vstack

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
h0 = np.array([1/np.sqrt(2),1/np.sqrt(2)])
h1 = np.array([1/np.sqrt(2),-1/np.sqrt(2)])
g0_daub = np.array([0.23037781,0.71484657,0.63088076,-0.02798376,-0.18703481,0.03084138,0.03288301,-0.0105940])
g0_sym = np.array([0.0322,-0.0126,-0.0992,0.2979,0.8037,0.4976,-0.0296,-0.0758])
mats = {'jpeg':jpegstd, 'zonal':zonal,'threshold':thresholdmask,'threshold1':thresholdmask1,'zonalbest':zonal_best}
wavelets = {'haar':'a'}


def encodewavelet(inputimage,waveletkernel,level):
    b = np.dot(inputimage,waveletkernel)
    for _ in range(level):
        b = np.dot(b.T,waveletkernel)
    return b

def haarkernel(img):
    m,_ = img.shape
    upperhaar=[]
    lowerhaar = []
    for i in range(0,m,2):
        q = np.zeros(m)
        p = np.zeros(m)
        q[i:i+len(h0)] = h0
        p[i:i+len(h1)] = h1
        upperhaar.append(q)
        lowerhaar.append(p)
    return np.vstack((upperhaar,lowerhaar)).T

def daubechieswavelet(inputimage,waveletkernel):
    m,_ = inputimage.shape
    upperhalf = []
    lowerhalf = []
    for i in range(0,m,len(g0_daub)):
        q = np.zeros(m)
        p = np.zeros(m)
        q[i:i+len(g0_daub)] =g0_daub
#         Reverse
        h0 = q[::-1]
        g1 = -1*h0[::2]
        for qq in range(h0):
            if qq %2 == 1:
                h0[qq] *= -1
#         p[i:i+len(g0_daub)] =
        

def main():
    args = parseArgs()
    if args.wavelet:
        kernel = haarkernel(args.inputimage)
        kernel = daubechieswavelet(args.inputimage,None)
        discrete_haar = encodewavelet(args.inputimage,kernel,3)
#         discrete_haar = discreteHaarWaveletTransform(args.inputimage)
        misc.imsave(args.o +'_haar.tif',discrete_haar)
#         print (np.eye(m/2, n) * h0).shape
#         upperhaar = np.vstack((np.eye(m/2, n) * h0),(np.eye(m/2, n) * h1))
#         print upperhaar
    encodedimgs = encode(args.inputimage,mats[args.quantmattype])
    decodedimg = decode(encodedimgs,args.inputimage.shape[0])
    if args.o:
        misc.imsave(args.o+'_reconstructed.tif',decodedimg)
    difference = args.inputimage - decodedimg
    if args.o:
        misc.imsave(args.o+'_diff.tif',difference)


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputimage', type=misc.imread)
    parser.add_argument('-quantmattype', choices=mats, default=mats.values()[0])
    parser.add_argument('-wavelet',choices=wavelets,default=wavelets.values()[0])
    parser.add_argument('-o',type=str,help='Outputfile')
    return parser.parse_args()



def decode(encodedimgs,rowlength):
    inverseimgs = inversetransform(encodedimgs)
    mergedimg = mergesubimages(inverseimgs,rowlength)
    return mergedimg
#     return readjustrange(mergedimg)
    
def mergesubimages(subimages,rowlength):
    ret = np.empty((rowlength,rowlength))
#     Actuially its 64 here
    for j in range(rowlength/(len(subimages)/rowlength)):
        for i in range(rowlength/(len(subimages)/rowlength)):
            subimg= subimages[(j*rowlength/(len(subimages)/rowlength))+i]
            for q in range(len(subimg)):
                a = j*len(subimg)+ q
                for p in range(len(subimg[0])):
                    b = i * len(subimg[0]) + p
                    ret[a,b] = subimg[q,p]   
    return ret
#     return subimages.reshape((512,512))
    
def inversetransform(encodedimgs):
    ret = []
    for img in encodedimgs:
        ret.append(idct2(img))
    return np.array(ret)

def encode(rawimg, quantmat):
#     adjustedimg = adjustrange(rawimg)
    rawimg = np.array(rawimg,dtype=float)
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
        quantized.append(dcttransform*quantmat)
    return np.array(quantized)

if __name__ == '__main__':
    main()
