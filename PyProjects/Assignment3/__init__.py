from argparse import ArgumentParser
from scipy import misc
import cmath
import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform

def main():
    args = parseargs()
    imgarr = misc.imread(args.inputimage)
    shape = imgarr.shape
    if args.shape:
        shape = (args.shape,args.shape)
    transformedimg = transform(imgarr,args.kernel,shape)
    if(args.o):
        misc.imsave(args.o, transformedimg)

def parseargs():
    parser = ArgumentParser()
    parser.add_argument('inputimage', help='The inputted image')
    parser.add_argument('-o', type=str, help='Output image path')
    parser.add_argument('-k','--kernel',choices=filters.keys(),default='gaussian',help='The kernel, which is going to be used')
    parser.add_argument('-s','--shape',type=int,help='Sqare shape of the kernel, default its the picture size')
    parser.add_argument('-sig','--sigma',type=float,help='Sigma / Cutoff frequency')
    return parser.parse_args()

def transform(imagearray,kernel,shape):
    dftrans = np.fft.fft2(imagearray)
    filt=filters[kernel]
    filteredimg = filt(dftrans, 2,shape)
    restoredimg = np.fft.ifft2(filteredimg)
    realimg = extractReal(restoredimg)
    return realimg

def extractReal(img):
    return np.array([[img[j][x].real for x in range(len(img[0]))] for j in range(len(img))])

def evenodd(x, y):
    if (x + y) % 2 == 0:
        return 1
    else:
        return -1

# Keep in mind that usually the datatype is uint8 so non negtaive values
def center(imagearr):
    copied = np.array(imagearr, dtype=int)
    for i in range(len(imagearr)):
        for j in range(len(imagearr[0])):
            copied[i][j] = imagearr[i][j] * evenodd(i, j)
    return copied

def idealfilter(dftarr,sigmasq,shape):
    kernel = ideal_kernel(shape=shape,sigma=sigmasq )
    ret = applykernel(kernel, dftarr)
    return ret


def ideal_kernel(shape=(3,3),sigma=1):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    g = np.sqrt( x * x + y * y )
    for i in range(len(g[0])):
        for j in range(len(g)):
            if g[i][j] <= sigma:
                g[i][j] = 1
            else:
                g[i][j] = 0
    gsum = g.sum()
    if gsum != 0:
        g /=gsum
    return g

def gauss2dkernel(shape=(3,3),sigmasq=1):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -((x*x)/m + (y*y)/n) / (2.*sigmasq) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
        
def butterworthkernel(shape=(3,3),sigmasq=1):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
# we use n=1 here
    g = 1. / (1 + (np.sqrt(x *x + y * y) / sigmasq) ** 2)
    gsum = g.sum()
    if gsum != 0:
        g /=gsum
    return g

def applykernel(kernel, arr):
    transformedimg = np.array(arr)
    x, y = kernel.shape
    print kernel
    offset = x / 2
    for i in range(offset, len(arr) - offset):
        for j in range(offset, len(arr[0]) - offset):
            weightedavg = 0
            for p in range(len(kernel)):
                for q in range(len(kernel)):
                    weightedavg += arr[i + (p - offset)][j + (q - offset)] * kernel[p][q]
            if weightedavg < 0:
                weightedavg = 0
            transformedimg[i][j] = weightedavg
    return transformedimg


def gaussianfilter(dftarr, sigmasq,shape):
#     Kernel has size 3x3
    kernel = gauss2dkernel(shape=shape,sigmasq=sigmasq )
    return applykernel(kernel, dftarr)

def butterworthfilter(dftarr, sigmasq,shape):
    kernel = butterworthkernel(shape, sigmasq)
    return applykernel(kernel, dftarr)


filters={
        'gaussian':gaussianfilter,
        'ideal' :idealfilter,
        'butterworth':butterworthfilter
         }
    

if __name__ == '__main__':
    main() 
