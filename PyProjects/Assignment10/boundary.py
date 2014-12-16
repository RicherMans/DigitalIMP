'''
Created on Nov 28, 2014

@author: richman
'''
'''
Note that in this exercise the given image seems to be binary, yet python
reads in a grayscale one, so we simply readjust it
'''

import argparse
from scipy import misc
import numpy as np

test = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0],
    [0, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
])

test1 = np.array([
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0],
                  [0, 1, 0, 1, 0, 0, 0],
                  [0, 1, 1, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  ])

ystep = [0, 1, 1, 1, 0, -1, -1, -1]
xstep = [-1, -1, 0, 1, 1, 1, 0, -1]

def main():
    chain = follow_boundary(test1)
    encode_chain(chain)
#     args = parseArgs()
# #     rescale image to binary
#     binimg = np.zeros((args.inputimage.shape))
#     binimg[args.inputimage < 255] = 1
#     follow_boundary((1,4), binimg)



def encode_chain(chain):
    codemap = {
               (1, 0):0,
               (1, 1):1,
               (0, 1):2,
               (-1, 1):3,
               (-1, 0):4,
               (-1, -1):5,
               (0, -1):6,
               (1, -1):7
               }
    
    for i in range(len(chain)):
        x, y = chain[i].b
        x1, y1 = chain[(i + 1) % len(chain)].b
        print codemap[(x1-x,y1-y)]
        




def follow_boundary(array):
    '''
    startpoint is a tuple, consisting of x,y variable to  indicate which is the startpoint of this boundary.
    returns an arr
    '''
    startpoint = findstartpoint(array)
    x, y = startpoint
#     that is b0
    b0 = (x, y)
    b1 = None
    b1_reached = False
    iterate = [Point(b0, (-1, 0))]
    foundpoints = []
    while(iterate and not b1_reached):
        curpoint = iterate.pop()
        bx, by = curpoint.b
        
        for i, (dx, dy) in enumerate(curpoint):
#             Case of b1 being not initialized, we need to save it!
            if (bx + dx, by + dy) == b1:
                b1_reached = True
            if array[bx + dx][by + dy] == 1 and not b1:
                b1 = (bx + dx, by + dy)
            if array[bx + dx][by + dy] == 1 and not b1_reached:
#                 Append the current neighbor of bx,by to be a valid Point and therefor found
                foundb = (bx + dx, by + dy)
                cx, cy = (curpoint)[i - 1]
#                 The offset is the new relative coordinate from the new found Point b* to the newly found background Point c*
                offx, offy = cx - dx, cy - dy
                foundc = (offx, offy)
#                 Create a new Point and store it in the iteration and the found array.
                foundpoint = Point(foundb, foundc)
                foundpoints.append(foundpoint)
                iterate.append(foundpoint)
                break
        if b1_reached:
            if (bx, by) == b0:
                return foundpoints
class Point():
    ''' A container for the given Points, stores 2 variables and returns an iterator and a given item
        The iterator returns for the current point the circle around it.
    '''
    neighboriter = zip(xstep, ystep)
    
    def __init__(self, b, c):
        self.b = b
        self.c = c
        
    def __repr__(self):
        ret = "".join(str(self.b))
        return ret
    
    def __iter__(self):
        index = self.neighboriter.index(self.c)
#         We need to check both of the slices, we get on one side the left over circle and on the other, the "togo"
        return iter(self.neighboriter[index:] + self.neighboriter[:index])
    
    def __getitem__(self, key):
        index = self.neighboriter.index(self.c)
        iterlist = self.neighboriter[index:] + self.neighboriter[:index]
        return iterlist[key]
#     During looping, the in operator can be used to check the b value
    def __eq__(self, other):
        return other == self.b


def findstartpoint(binimg):
    x, y = np.where(binimg == binimg.max())
# The problem for choosing the uppermost left most Point, is choosing the x,y
# pair which has the lowest indices as values

    ind = (x + y).argmin()
#     Return the first occurance of the lowest value
    return (x[ind], y[ind])
    

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputimage', type=misc.imread)
    return parser.parse_args()


if __name__ == '__main__':
    main()
