from argparse import ArgumentParser
from scipy import misc, random
import cmath
import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform
from numpy import real, matlib
from numpy.matlib import randn
import argparse
import os



def main():
    args = parseargs()
    imagearr = misc.imread(args.inputimage)
    noisedimg = addNoise(args.noise, imagearr)
    writeImg(noisedimg, 'uniform', 'orig')
#     alphatrimmedMeanFilter(noisedimg, 2)
#     writeImg(noisedimg, args.noise, 'OriginalNoise')
    filterImgWriteOut(args,noisedimg)
        

def filterImgWriteOut(args,img):
    if args.arithmetic:
        arithmeticimg = arithmeticMeanFilter(img,args.size)
        writeImg(arithmeticimg, args.noise, 'arithmetic')
    elif args.geometric:
        geometricimg = geometricMeanFilter(img,args.size)
        writeImg(geometricimg, args.noise, 'geometric')
    elif args.harmonic:
        harmonicimg = harmonicMeanFilter(img,args.size)
        writeImg(harmonicimg, args.noise, 'harmonic')
    elif args.charmonic:
        charmonicimg = contraharmonicMeanFilter(img,1,args.size)
        writeImg(charmonicimg, args.noise, 'counterharmonic')
    elif args.median:
        medianimg = medianFilter(img,args.size)
        writeImg(medianimg, args.noise, 'median')
    elif args.max:
        maximg = maxFilter(img,args.size)
        writeImg(maximg, args.noise, 'max')
    if not any([args.arithmetic,args.geometric,args.harmonic,args.charmonic,args.median]):
        arithmeticimg = arithmeticMeanFilter(img,args.size)
        geometricimg = geometricMeanFilter(img,args.size)
        medianimg = medianFilter(img,args.size)
        harmonicimg = harmonicMeanFilter(img,args.size)
        charmonicimg = contraharmonicMeanFilter(img,1,args.size)
        maximg = maxFilter(img,args.size)
        midpimg = midpointFilter(img, args.size)
        
        writeImg(arithmeticimg, args.noise, 'arithmetic')
        writeImg(geometricimg, args.noise, 'geometric')
        writeImg(medianimg, args.noise, 'median')
        writeImg(harmonicimg, args.noise, 'harmonic')
        writeImg(charmonicimg, args.noise, 'counterharmonic')
        writeImg(maximg, args.noise, 'max')
        writeImg(midpimg, args.noise, 'midpoint')


def writeImg(img,noisetype,filtertype):
    misc.imsave(noisetype+'_'+filtertype+'.tif', img)    
    print "Wrote %s "%(noisetype+'_'+filtertype+'.tif')
    
def addNoise(noise,imagearr):
    if noise == 'gaussian':
        curnoise= gaussiannoise(255/2, 255/8, imagearr)
    elif noise == 'uniform':
        curnoise = uniformnoise(imagearr,0, 255)
    noisedimg = imagearr + curnoise
    normalize(noisedimg)
    return noisedimg

def uniformnoise(data,low=0,high=1):
    return np.random.uniform(low,high,size=data.shape)

def gaussiannoise(mean,var,data):
    return np.random.normal(mean,var,data.shape)

def normalize(arr):
    for i in range(len(arr)):
        normalizer = max(arr[i])
        for j in range(len(arr[0])):
            arr[i][j] = arr[i][j]/normalizer * 255
            
def arithmeticMeanFilter(arr,shape=(3,3)):
    x,y = shape
    transformedimg = np.copy(arr)
    xoff = x/2
    yoff = y/2
    for i in range(xoff,len(arr)-xoff):
        for j in range(yoff,len(arr[0])-yoff):
            average = 0
            for p in range(x):
                for q in range(y):
                    average += arr[i+(p-xoff)][j+(q-yoff)]
            if average<0:
                average = 0
            transformedimg[i][j] = average/(x*y)
    return transformedimg
            
def geometricMeanFilter(arr,shape=(3,3)):
    x,y = shape
    xoff = x/2
    yoff = y/2
    transformedimg = np.copy(arr)
    for i in range(xoff,len(arr)-xoff):
        for j in range(yoff,len(arr[0])-yoff):
            average = 0
            for p in range(x):
                for q in range(y):
                    average *= arr[i+(p-xoff)][j+(q-yoff)]
            if average<0:
                average = 0
            transformedimg[i][j] = math.pow(average,1./x*y)
    return transformedimg

def harmonicMeanFilter(arr,shape=(3,3)):
    x,y = shape
    xoff = x/2
    yoff = y/2
    transformedimg = np.copy(arr)
    for i in range(xoff,len(arr)-xoff):
        for j in range(yoff,len(arr[0])-yoff):
            average = 0
            for p in range(x):
                for q in range(y):
                    average += 1./arr[i+(p-xoff)][j+(q-yoff)]
            if average<0:
                average = 0
            transformedimg[i][j] = (x*y)/average
    return transformedimg
            
def contraharmonicMeanFilter(arr,q,shape=(3,3)):
    x,y = shape
    xoff = x/2
    yoff = y/2
    transformedimg = np.copy(arr)
    for i in range(xoff,len(arr)-xoff):
        for j in range(yoff,len(arr[0])-yoff):
            harmonicpart = 0
            contraharmonicpart = 0
            for p in range(x):
                for q in range(y):
                    harmonicpart += np.power(arr[i+(p-xoff)][j+(q-yoff)],q)
                    contraharmonicpart += np.power(arr[i+(p-xoff)][j+(q-yoff)],q+1)
            transformedimg[i][j] = contraharmonicpart/harmonicpart
    return transformedimg

def medianFilter(arr,shape=(3,3)):
    x,y = shape
    xoff = x/2
    yoff = y/2
    transformedimg = np.copy(arr)
    for i in range(xoff,len(arr)-xoff):
        for j in range(yoff,len(arr[0])-yoff):
            maxpart = 0
            tmpmax = 0
            for p in range(x):
                for q in range(y):
                    tmpmax = arr[i+(p-xoff)][j+(q-yoff)]
                    if tmpmax > maxpart:
                        maxpart = tmpmax
            transformedimg[i][j] = maxpart
    return transformedimg

def maxFilter(arr,shape=(3,3)):
    x,y = shape
    xoff = x/2
    yoff = y/2
    transformedimg = np.copy(arr)
    for i in range(xoff,len(arr)-xoff):
        for j in range(yoff,len(arr[0])-yoff):
            maxpart = 0
            tmpmax = 0
            for p in range(x):
                for q in range(y):
                    tmpmax = arr[i+(p-xoff)][j+(q-yoff)]
                    if tmpmax > maxpart:
                        maxpart = tmpmax
            transformedimg[i][j] = maxpart
    return transformedimg

def midpointFilter(arr,shape=(3,3)):
    x,y = shape
    xoff = x/2
    yoff = y/2
    transformedimg = np.copy(arr)
    for i in range(xoff,len(arr)-xoff):
        for j in range(yoff,len(arr[0])-yoff):
            maxpart = 0
            tmp = 0
            minpart = 1000
            for p in range(x):
                for q in range(y):
                    tmp = arr[i+(p-xoff)][j+(q-yoff)]
                    if tmp > maxpart:
                        maxpart = tmp
                    if tmp < minpart:
                        minpart = tmp
            transformedimg[i][j] = 0.5 *(maxpart +minpart)
    return transformedimg

def alphatrimmedMeanFilter(arr,d,shape=(3,3)):
    x,y = shape
    xoff = x/2
    yoff = y/2
    transformedimg = np.copy(arr)
    for i in range(xoff,len(arr)-xoff):
        for j in range(yoff,len(arr[0])-yoff):
            tmp = 1000
            tmpcp = np.copy(arr[i:i+x-xoff][j:j+y-yoff])
            print tmpcp
            for p in range(x):
                for q in range(y):
                    tmp += arr[i+(p-xoff)][j+(q-yoff)]
            transformedimg[i][j] = 1./(x*y - d)
    return transformedimg

noises={
       'gaussian',
       'uniform'
       }    

filters = {
           'artihmetic':arithmeticMeanFilter,
           'geometric':geometricMeanFilter,
           'median':medianFilter,
           'harmonic':harmonicMeanFilter,
           'charmonic':contraharmonicMeanFilter
           }

def shape(n):
    try:
        x,y = n.split("#")
        x = int(x)
        y = int(y)
        return x,y
    except:
        raise argparse.ArgumentError('Wrong arguments given, shape should be 1#1')

def parseargs():
    parser = ArgumentParser()
    parser.add_argument('inputimage', help='The inputted image')
    parser.add_argument('-o', type=str, help='Output image/s path')
    parser.add_argument('-n','--noise',choices=noises,default='gaussian')
    parser.add_argument('--size',type=shape,help='Size of the neighborhood mask',default=(3,3))
    parser.add_argument('--arithmetic',action='store_true')
    parser.add_argument('--geometric',action='store_true')
    parser.add_argument('--median',action='store_true')
    parser.add_argument('--harmonic',action='store_true')
    parser.add_argument('--max',action='store_true')
    parser.add_argument('--midpoint',action='store_true')
    parser.add_argument('--charmonic',type=int,help='Uses the contraharmonic filtering. Needs to filter out')
    return parser.parse_args()
    

if __name__ == '__main__':
    main() 