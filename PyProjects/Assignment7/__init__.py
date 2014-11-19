#===============================================================================
# In this assignment it is necessary to implement Morphological processing
#===============================================================================
import argparse
from scipy import misc, fftpack
import cmath
import numpy as np

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
thresholdmask = np.array([
                 [8, 7, 6, 5, 4, 3, 2, 1],
                 [7, 6, 5, 4, 3, 2, 1, 0],
                 [6, 5, 4, 3, 2, 1, 0, 0],
                 [5, 4, 3, 2, 1, 0, 0, 0],
                 [4, 3, 2, 1, 0, 0, 0, 0],
                 [3, 2, 2, 1, 0, 0, 0, 0],
                 [2, 1, 0, 0, 0, 0, 0, 0],
                 [1, 0, 0, 0, 0, 0, 0, 0]
                 ])


mats = {'jpeg':jpegstd, 'zonal':zonal,'threshold':thresholdmask}

def main():
    args = parseArgs()
#     rr = np.random.normal(100, 100, size=(16, 16))
#     encodedimgs = encode(rr, args.quantmattype)
    encodedimgs = encode(args.inputimage,args.quantmattype)
    decodedimg = decode(encodedimgs)
    if args.o:
        misc.imsave(args.o+'_reconstructed.tif',decodedimg)
    difference = args.inputimage - decodedimg
    if args.o:
        misc.imsave(args.o+'_diff.tif',difference)

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputimage', type=misc.imread)
    parser.add_argument('-quantmattype', choices=mats, default=mats.values()[0])
    parser.add_argument('-o',type=str,help='Outputfile')
    return parser.parse_args()


def readjustrange(mergedimg):
    adjustedimg = np.array(mergedimg,dtype=float)
    adjustedimg += 128
    return adjustedimg


def decode(encodedimgs):
    inverseimgs = inversetransform(encodedimgs)
    mergedimg = mergesubimages(inverseimgs)
    return readjustrange(mergedimg)
    
def mergesubimages(subimages):
    return subimages.reshape((512,512))
    
def inversetransform(encodedimgs):
    ret = []
    [ret.append(fftpack.idct(img)) for img in encodedimgs]
    return np.array(ret)

def adjustrange(rawimg):
    '''
    This function just transforms the input image to be in the range of
    [-127,128] instead of [0,255]
    '''
    adjustedrangeimg = np.array(rawimg, dtype=np.float)
    adjustedrangeimg -= 128
    return adjustedrangeimg
    
def encode(rawimg, quantmat):
    adjustedimg = adjustrange(rawimg)
    subimages = np.array(list(constructsubimages(adjustedimg)))
    dcttransformedimgs = forwardtransform(subimages)
    inverseimgs = inversetransform(dcttransformedimgs)
    print inverseimgs[0]
    print subimages[0]
    quantimgs = quantitize(dcttransformedimgs, quantmat)
    return quantimgs
    
    
def constructsubimages(img, size=8):
    for i in range(0, len(img), size):
        for j in range(0, len(img[0]), size):
            yield img[i:i + size, j:j + size]

def forwardtransform(subimages):
    dctimages = []
    for image in subimages:
        dctimages.append(fftpack.dct(fftpack.dct(image.T)).T)
    return np.array(dctimages)
        
def quantitize(dcttransformedimgs, quantmat):
    quantized = []
    for dcttransform in dcttransformedimgs:
        quantized.append(np.round(np.multiply(dcttransform, quantmat)))
    return np.array(quantized)

if __name__ == '__main__':
    main()
