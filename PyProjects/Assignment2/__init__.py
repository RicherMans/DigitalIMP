from argparse import ArgumentParser
import numpy as np
from scipy import misc

def main():
    args=parseArgs()
    inputimage = misc.imread(args.inputimage)
    mask = np.array([[0,-1,0],[-1,4+args.a,-1],[0,-1,0]])
    outputimage = transform(inputimage,mask)
    writeimage(outputimage,args.o)

def writeimage(binaryimage,outpath):
    misc.imsave(outpath, binaryimage)

def transform(arr,mask):
    transformedimg = np.array(arr)
    for i in range(1,len(arr)-1):
        for j in range(1,len(arr[0])-1):
            weightedavg = 0
            for p in range(len(mask)):
                for q in range(len(mask)):
                    weightedavg += transformedimg[i+(p-1)][j+(q-1)] * mask[p][q]
            if weightedavg<0:
                weightedavg = 0
            transformedimg[i][j] = weightedavg
    return transformedimg

def parseArgs():
    parser = ArgumentParser()
    parser.add_argument('inputimage',help='The inputted image')
    parser.add_argument('-o',type=str,help='Output image path')
    parser.add_argument('-a',type=float,help='The centered added value A',default=0)
    return parser.parse_args()




if __name__ == '__main__':
    main()