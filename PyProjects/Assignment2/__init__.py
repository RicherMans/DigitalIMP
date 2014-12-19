from argparse import ArgumentParser
import numpy as np
from scipy import misc
from Assignment4 import shape


def main():
    args = parseArgs()
    inputimage = misc.imread(args.inputimage)
    laplacian_mask = np.array([[-1, -1, -1], [-1, 8+args.a, -1], [-1, -1, -1]])
    laplacian_img = transform(inputimage, laplacian_mask)
    writeimage(laplacian_img, 'Laplacian_' + args.o)
    sharpenedimg = laplacian_img + inputimage
    writeimage(sharpenedimg, 'Sharpened_' + args.o)
    sobelimg = transformsobel(inputimage)
    writeimage(sobelimg, 'Sobel_' + args.o)
    smoothedimg = transform(sobelimg, np.ones((5, 5)),True)
    writeimage(smoothedimg, 'smooth_' + args.o)
    maskedimg = scale(smoothedimg * laplacian_img)
    writeimage(maskedimg, 'maskedimg_' + args.o)
    finalsharpimg = powerlaw(maskedimg + inputimage, 0.5)
    writeimage(finalsharpimg, 'final_' + args.o)


def scale(arr):
    min_val = np.min(arr)
    scaled = arr-min_val
    scaled[scaled <=np.finfo(np.float32).eps] = 0.
    max_value= np.max(scaled)
    if max_value != 0. : scaled= scaled*(255./max_value)
    return scaled

def powerlaw(inputimg, gamma):
    # normalize(inputimg)
    return scale(1 * np.power(inputimg, gamma))


def writeimage(binaryimage, outpath):
    misc.imsave(outpath, binaryimage)


def transformsobel(arr):
    def sobel():
        return (np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))
    xkernl, ykernl = sobel()
    gx = transform(arr, xkernl,False)
    gy = transform(arr, ykernl,False)
    return scale(np.hypot(gx, gy))


def transform(arr, mask,doscale=True):
    transformedimg = np.array(arr, dtype=int)
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            weightedavg = 0
            for p in range(len(mask)):
                for q in range(len(mask)):
                    weightedavg += arr[(i +
                                        (p - 1)) % len(arr)][(j + (q - 1)) % len(arr[0])] * mask[p][q]
            transformedimg[i][j] = weightedavg
    if doscale:
        return scale(transformedimg)
    else:
        return transformedimg


def parseArgs():
    parser = ArgumentParser()
    parser.add_argument('inputimage', help='The inputted image')
    parser.add_argument('-o', type=str, help='Output image path')
    parser.add_argument(
        '-a', type=float, help='The centered added value A', default=0)
    return parser.parse_args()


if __name__ == '__main__':
    main()
