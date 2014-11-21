'''
Created on 21 Nov, 2014

@author: hedi7
'''
import argparse
from scipy import misc
import numpy as np

mask = np.array([
                [1,1,1],
                [1,1,1],
                [1,1,1]
                ]
                )
def main():
    args = parseArgs()
    eros = erosion(args.inputimage)
    if args.o:
        misc.imsave(args.o+'_erosion.jpg', eros)
    
def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputimage', type=misc.imread)
    parser.add_argument('-o',type=str,help='Outputfile')
    return parser.parse_args()

def erosion(inputimg):
    maskoffset = len(mask)/2
    oldx,oldy = inputimg.shape
    returnimg = np.ones(inputimg.shape)
    for i in range(maskoffset,len(inputimg)-maskoffset):
        for j in range(maskoffset,len(inputimg[0])-maskoffset):
            found = False
            for p in range(len(mask)):
                for q in range(len(mask)):
                    if inputimg[i+(p-maskoffset)][j+(q-maskoffset)] != mask[p][q]:
                        found = True
                        break
#                         Centered pixel of the matrix
            if found :
                returnimg[i,j] = 0
#    Slice up the resulting image since the borders do not fit into the mask 
    return returnimg[maskoffset:oldx-maskoffset,maskoffset:oldy-maskoffset]
                    
#             if inputimg[i,j] != mask[maskoffset,maskoffset]:
#                         Retain the contered element
    return returnimg

if __name__ == '__main__':
    main()