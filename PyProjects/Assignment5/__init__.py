import argparse
from scipy import misc
from scipy.ndimage.filters import convolve as convolveim
from scipy.ndimage import gaussian_filter
import numpy as np
import sys
from numpy import real

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputimage', type=misc.imread)
    parser.add_argument('-a',help='the parameter a in the equation, which is equal to b',default=0.1,type=float)
    parser.add_argument('-T',help='the parameter T',default=1,type=float)
#     parser.add_argument('-o',help='outputs the produced image')
    return parser.parse_args()


def main():
    args = parseArgs()
    blurredkernel = blurringkernel(args.inputimage.shape,args.T,args.a,args.a)
    motionblurredimg = applymotionblur(args.inputimage,blurredkernel,blurringfilter)
#     normalize(motionblurredimg)
    misc.imsave('blurred.jpg',motionblurredimg )
    blur_noise_img = motionblurredimg + gaussiannoise(0, 650, motionblurredimg)
    misc.imsave('blurred_w_gauss.jpg',blur_noise_img)
    restored_inverse = filterimg(args.inputimage,blur_noise_img, blurredkernel, inversefilter)
    misc.imsave('restored_inversefilt.jpg',restored_inverse)
    restored_wiener = filterimg(args.inputimage, blur_noise_img, blurredkernel, wienerfilter)
    misc.imsave('restored_wiener_param1.jpg',restored_wiener)
#     restored_wiener_2 =filterimg(args.inputimage, blur_noise_img, blurredkernel, wienerfilter,2.)
#     misc.imsave('restored_wiener_param2.jpg', restored_wiener_2)
#     restored_wiener_10 =filterimg(args.inputimage, blur_noise_img, blurredkernel, wienerfilter,10.)
#     misc.imsave('restored_wiener_param10.jpg', restored_wiener_10)
#     restored_wiener_011 =filterimg(args.inputimage, blur_noise_img, blurredkernel, wienerfilter,.01)
#     misc.imsave('restored_wiener_param011.jpg', restored_wiener_011)
    
def gaussiannoise(mean,var,data):
    return np.random.normal(mean,var,data.shape)

def eucl_dist(i,j,M,N):
    return np.sqrt((i-(M/2))**2+(j-(N/2))**2)

def normalize(arr):
    for i in range(len(arr)):
        normalizer = max(arr[i])
        for j in range(len(arr[0])):
            arr[i][j] = arr[i][j]/normalizer * 255

def blurringkernel(shape,T,a,b):
    xx,yy = shape
    x, y = np.mgrid[0.:xx, 0.:yy]
    x[0,0]=.1
#     d[((len(x))/2)-1][(len(y)/2)-1] = .1111
#     return (T/(np.pi*((a+b)*d)))*np.sin(np.pi*(d*(a+b))) * np.exp(-1j*np.pi*((a+b)*d))
    return (T/(np.pi*(x*a+y*b))) * np.sin(np.pi*(x*a+y*b)) * np.exp(-1j*np.pi*(x*a+y*b))

def blurringfilter(dftarr, kernel):
    x = []
    for i in kernel:
        x.append(max(i))
    print max(x)
    print kernel[0][0]
#     print kernel[len(kernel)/2][len(kernel)/2]
    return applykernel(dftarr,kernel)

def powerspectrum(img):
    ret = img*img
    return ret


def wienerkernel(origimg,noise,blurredkernel,param):
    '''
    Calculates the wiener kernel, which is H*/(H^2+param*(S_n/S_f))
    '''
    h = blurredkernel
#     Power spectrum of noise
#     s_n= powerspectrum(noise)
#     Power spectrum of original image
#     s_f = powerspectrum(origimg)
    s_n =1.
    s_f = 100.
    return np.conj(h)/((h*h)+param*(s_n/s_f))

def wienerfilter(dftarr,noise,blurredkernel,param=1):
    kernel = wienerkernel(dftarr,noise,blurredkernel,param)
    return applykernel(dftarr, kernel)

'''
Inverse filter is calculated using F^ = G(u,v) H(u,v), since we know H(u,v) , which is the blurring function
we can simply return the inverse of H(u,v)
'''
def inversekernel(kernel):
    return 1./kernel

def inversefilter(dftarr,noise, kernel):
    kernel = inversekernel(kernel)
    '''
    returns F(u,v) + N(u,v) * H^-1 (u,v)
    '''
    return dftarr+applykernel(noise, kernel)

def applykernel(arr,kernel):
    transformedimg = np.copy(arr)
    transformedimg *= kernel
    return transformedimg

def extractReal(img):
    return np.array([[(img[j][i]).real for i in range(len(img))] for j in range(len(img[0]))])

def center(imagearr):
    m,n = imagearr.shape
    x,y = np.ogrid[0:m,0:n]
    return imagearr * ((-1)**(x+y))

def applymotionblur(imagearray,kernel,filt,*args):
    imagearray= center(imagearray)
    dftrans = np.fft.fft2(imagearray)
    filteredimg = filt(dftrans, kernel,*args) 
    inversefft= np.fft.ifft2(filteredimg)
    realimg = extractReal(inversefft)
    return center(realimg)
#     return realimg
def filterimg(origimg,noise,blurkernel,filt,*args):
    imagearray = center(origimg)
    noisearr = center(noise)
    dfttrans = np.fft.fft2(imagearray)
    noisedft = np.fft.fft2(noisearr)
    filterredimg = filt(dfttrans,noisedft,blurkernel,*args)
    inversefft= np.fft.ifft2(filterredimg)
    realimg = extractReal(inversefft)
    return center(realimg)
    

if __name__ == '__main__':
    main()