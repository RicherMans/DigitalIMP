from argparse import ArgumentParser
from scipy import misc
import cmath
import numpy as np
import math
from matplotlib import pyplot as plt
from numpy import real, convolve
import scipy

def main():
    args = parseargs()
    imgarr = misc.imread(args.inputimage)
    shape = imgarr.shape
    transformedimg = transform(imgarr, args.kernel, shape ,args.sigma)
#     print transformedimg
#     print transformedimg[0][0]
    if(args.o):
        misc.imsave(args.o, transformedimg)
        
        
def parseargs():
    parser = ArgumentParser()
    parser.add_argument('inputimage', help='The inputted image')
    parser.add_argument('-o', type=str, help='Output image path')
    parser.add_argument('-k', '--kernel', choices=filters.keys(), default='gaussian', help='The kernel, which is going to be used')
    parser.add_argument('-sig', '--sigma', type=float, help='Sigma / Cutoff frequency')
    return parser.parse_args()

def transform(imagearray, kernel, shape,sigma):
    imagearray=center(imagearray)
    dftrans = np.fft.fft2(imagearray)
    filt = filters[kernel]
    filteredimg = filt(dftrans,sigma**2 , shape)
    restoredimg = np.fft.ifft2(filteredimg)
    realimg = extractReal(restoredimg)
    return center(realimg)
#     return realimg

def extractReal(img):
    return np.array([[img[j][i].real for i in range(len(img[0]))] for j in range(len(img))])

# Keep in mind that usually the datatype is uint8 so non negtaive values
def center(imagearr):
    m,n = imagearr.shape
    x,y = np.ogrid[0:m,0:n]
    return imagearr * ((-1)**(x+y))

def idealfilter(dftarr, sigmasq, shape):
    kernel = ideal_kernel(shape=shape, sigma=sigmasq)
    return applykernel(kernel,dftarr)


def ideal_kernel(shape, sigma):
    xx,yy =shape
#     m, n = [(ss -1) / 2. for ss in shape]
    x, y = np.mgrid[0:xx, 0:yy]
    d = eucl_dist(x, y, len(x), len(y))
    for i in range(len(d[0])):
        for j in range(len(d)):
            if d[i][j] <= sigma:
                d[i][j] = 1
            else:
                d[i][j] = 0
    gsum = d.sum()
    if gsum != 0:
        d /= gsum
    return d

def eucl_dist(i,j,M,N):
    return np.sqrt((i-(M/2))**2+(j-(N/2))**2)

def gauss2dkernel(shape, sigma):
    xx,yy =shape
#     m, n = [(ss -1) / 2. for ss in shape]
    x, y = np.mgrid[0:xx, 0:yy]
    d = eucl_dist(x, y, len(x), len(y))
    h = np.exp(-(d**2/(2.*sigma)))
#     h[ h < np.finfo(h.dtype).eps * h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
        
def butterworthkernel(shape, sigmasq):
    xx,yy =shape
    x, y = np.mgrid[0:xx, 0:yy]
    d = eucl_dist(x, y, len(x), len(y))
# we use n=1 here
    g = 1. / (1 + (np.sqrt(d**2) / sigmasq) ** 2)
    gsum = g.sum()
    if gsum != 0:
        g /= gsum
    return g

def applykernel(kernel, arr):
    return arr*kernel


def blurringfilter(dftarr, sigmasq, shape):
    kernel = gauss2dkernel(shape=shape, sigma=sigmasq)
    return applykernel(kernel, dftarr)

def butterworthfilter(dftarr, sigmasq, shape):
    kernel = butterworthkernel(shape, sigmasq)
    return applykernel(kernel, dftarr)


filters = {
        'gaussian':blurringfilter,
        'ideal' :idealfilter,
        'butterworth':butterworthfilter
         }
    

if __name__ == '__main__':
    main() 
