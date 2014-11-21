#===============================================================================
# In this assignment it is necessary to implement Compression
#===============================================================================
import argparse
from scipy import misc, fftpack
from scipy.fftpack import dct
import cmath
import numpy as np
from matplotlib import pyplot

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

haar = []
mats = {'jpeg':jpegstd, 'zonal':zonal,'threshold':thresholdmask,'threshold1':thresholdmask1,'zonalbest':zonal_best}
wavelets = {'haar':haar}

def main():
    args = parseArgs()
    encodedimgs = encode(args.inputimage,mats[args.quantmattype])
    decodedimg = decode(encodedimgs,args.inputimage.shape[0])
    if args.o:
        misc.imsave(args.o+'_reconstructed.tif',decodedimg)
    difference = args.inputimage - decodedimg
    if args.o:
        misc.imsave(args.o+'_diff.tif',difference)

def parseArgs():
#     x=np.zeros(shape=(8,8))
#     p = 0
#     for i in reversed(xrange(len(x))):
#         for j in range(i-p):
#             x[len(x)-i-1,j] = 1
#     print x
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
