from argparse import ArgumentParser
from scipy import misc, ndimage
import cmath
import numpy as np
import math
from matplotlib import pyplot as plt
from numpy.fft.fftpack import fft2 as f2, ifft2
from numpy.fft import fftpack
from numpy.linalg import linalg
def main():
    args = parseargs()
    imgarr = misc.imread(args.inputimage)
    transformedimg = transform(imgarr)
    if(args.o):
        misc.imsave(args.o,transformedimg)
    

def parseargs():
    parser = ArgumentParser()
    parser.add_argument('inputimage', help='The inputted image')
    parser.add_argument('-o', type=str, help='Output image path')
    return parser.parse_args()

def transform(imagearray):
    zeropaddimg = center(imagearray)
    dftrans = fft2(zeropaddimg)
    filteredimg=gaussianfilter(dftrans, 1)
    restoredimg = ifft2(filteredimg)
    return center(restoredimg)

def evenodd(x,y):
    if (x+y)%2 == 0:
        return 1
    else:
        return -1

# Keep in mind that usually the datatype is uint8 so non negtaive values
def center(imagearr):
    copied = np.array(imagearr,dtype=int)
    for i in range(len(imagearr)):
        for j in range(len(imagearr[0])):
            copied[i][j] = imagearr[i][j] * evenodd(i, j)
    return copied


def idealfilter(dftarr):
    
    pass

def butterworthfilter(dftarr,cutoff):
    pass

def gaussian_kernel(size, sigmasq,size_y=None):
    size = int(size)
    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)
    x, y = np.mgrid[-size:size+1, -size_y:size_y+1]
    g = np.exp(-(x**2/float(size)+y**2/float(size_y))/2.*sigmasq)
    return g / g.sum()

def distanceToCenter(x,y,centerx,centery):
    return np.sqrt((x-centerx)**2 + (y-centery)**2)

def ideal_kernel(size,distance,size_y=None):
    size = int(size)
    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)
    x, y = np.mgrid[-size:size+1, -size_y:size_y+1]
    center_x = x/2
    center_y = y/2
    g = distanceToCenter(x, y, center_x, center_y)
    

def applykernel(kernel,arr):
    transformedimg = np.array(arr)
    x,y = kernel.shape
    offset = x/2
    for i in range(offset,len(arr)-offset):
        for j in range(offset,len(arr[0])-offset):
            weightedavg = 0
            for p in range(len(kernel)):
                for q in range(len(kernel)):
                    weightedavg += transformedimg[i+(p-offset)][j+(q-offset)] * kernel[p][q]
            if weightedavg<0:
                weightedavg = 0
            transformedimg[i][j] = weightedavg
    return transformedimg

def gaussianfilter(dftarr,sigmasq):
#     Kernel has size 3x3
# return exp(-norm(v1-v2, 2)**2/(2.*sigma**2))
    kernel = gaussian_kernel(2,sigmasq=2.5)
    appliedimg = applykernel(kernel,dftarr)
    return appliedimg
def butterworth(dftarr,cutoff):
    pass



# def applyfilter(dftarr,filter='butterworth'):
#     if filter=='butterworth':
#         pass
#     elif filter =='gaussian':
#     elif filter=='ideal':
#         pass

def idft(t):
    x = []
    N = len(t)
    for n in range(N):
        a = 0
        for k in range(N):
            a += t[k]*cmath.exp(2j*cmath.pi*k*n*(1./N))
        a /= N
        x.append(a)
    return x

def fft2(img):
    N,M = img.shape
    return 1./N*M * f2(img)

def dft(img):
    cpimg = np.array(img)
    N,M = img.shape
    for u in range(N):
        for v in range(M):
            a = 0
            for x in range(N):
                for y in range(M):
                    a += img[x][y]*cmath.exp(-2j*cmath.pi*((u*x*1./N) + (v*y*1./M)))
            cpimg[u,v] = a 
    return cpimg

    

if __name__ == '__main__':
    main() 
