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
                  [0,0,0,0,0,0,0],
                  [0,0,0,1,0,0,0],
                  [0,0,1,0,1,0,0],
                  [0,0,1,0,0,0,0],
                  [0,1,0,1,0,0,0],
                  [0,1,1,1,0,0,0],
                  [0,0,0,0,0,0,0],
                  ])


def main():
    follow_boundary((2, 3), test1)
# args = parseArgs()
# rescale image to binary
# binimg = np.zeros((args.inputimage.shape))
# binimg[args.inputimage > 1] = 1
# boundary_follow(binimg)


def encode_chain(chain):
    for i in range(len(chain)):
        x, y = chain[i]
        x1,y1 = chain[i+1]


def follow_boundary(startpoint, array):
    '''
    startpoint is a tuple, consisting of x,y variable to  indicate which is the startpoint of this boundary.
    returns an arr
    '''
    x, y = startpoint
    ystep = [0, 1, 1, 1, 0, -1, -1, -1]
    xstep = [-1, -1, 0, 1, 1, 1, 0, -1]
    
#     that is b0
    iterate = [point((x,y),(-1,0))]
#     iterate= [(x, y),tuple(zip(xstep,ystep))]
    
    foundpoints=[]
    b0 = (x, y)
    b1 = None
    b1_reached = False
    while(iterate and not b1_reached):
#         bx, by,(xstep,ystep) = iterate.pop()
        curpoint = iterate.pop()
        bx,by = curpoint.b
        
        for i,(dx,dy) in enumerate(curpoint):
#             Case of b1 being not initialized, we need to save it!
            if (bx + dx,by + dy)==b1:
                b1_reached= True
            if array[bx + dx][by + dy] == 1 and not b1:
                b1=(bx + dx,by + dy)
            if array[bx + dx][by + dy] == 1:
#                 Append the current neighbor of bx,by to be a valid point and therefor found
                foundb = (bx+dx,by+dy)
                cx,cy = (curpoint)[i-1]
#                 The offset is the new relative coordinate from the new found point b* to the newly found background point c*
                offx,offy = cx-dx,cy-dy
                foundc = (offx,offy)
#                 Create a new point and store it in the iteration and the found array.
                foundpoint = point(foundb,foundc)
                print [i for i in curpoint]
                print "for %i %i with c : %s point found: %s"%(bx,by,foundc,foundb)
                found = np.copy(array)*255
                found[foundb] = 2
                cx,cy = foundc
                found[bx+cx,by+cy] = 1
                print found 
                foundpoints.append(foundpoint)
                iterate.append(foundpoint)
                break
        print bx,by
        if b1_reached:
            if (bx,by) == b0:
                print foundpoints
#     found = np.zeros(test.shape)
#     for corr in foundpoints:
#         x, y = corr.b
#         found[x, y] = 1
#     print found

class point():
    
    xstep = [-1, -1, 0, 1, 1, 1, 0, -1]
    ystep = [0, 1, 1, 1, 0, -1, -1, -1]
    
    neighboriter = zip(xstep,ystep)
    
    def __init__(self,b,c):
        self.b = b
        self.c = c
        
    def __repr__(self):
        ret = " ".join(str(self.b))
        return ret
    
    def __iter__(self):
        index= self.neighboriter.index(self.c )
#         We need to check both of the slices, we get on one side the left over circle and on the other, the "togo"
        return iter(self.neighboriter[index:]+self.neighboriter[:index])
    
    def __getitem__(self,key):
        index= self.neighboriter.index(self.c)
        iterlist = self.neighboriter[index:]+self.neighboriter[:index]
        return iterlist[key]
#     During looping, the in operator can be used to check the b value
    def __eq__(self, other):
        return other == self.b

def boundary_follow(binimg):
    x, y = np.where(binimg == 1)
# The problem for choosing the uppermost left most point, is choosing the x,y
# pair which has the lowest indices as values
    xy_pair = np.sqrt(x.argmin() ** 2 + y[x.argmin()] ** 2)
    yx_pair = np.sqrt(y.argmin() ** 2 + x[y.argmin()] ** 2)
    b0 = 0
    c0 = 0
    if xy_pair < yx_pair:
        b0 = (x[y.argmin()], y.argmin())
        c0 = (x[y.argmin()] - 1, y.argmin())
    else:
        b0 = (x.argmin(), y[x.argmin()])
        c0 = (x.argmin() - 1, y[x.argmin()])
    sx, sy = b0
    b1 = 0
    foundbs = []
    foundcs = []
    ystep = [0, 1, 1, 1, 0, -1, -1, -1]
    xstep = [-1, -1, 0, 1, 1, 1, 0, -1]
    for x in range(sx, len(binimg) - 1):
        for y in range(sy, len(binimg[0]) - 1):
            b = 0
            c = 0
            for dx, dy in zip(xstep, ystep):
                if binimg[x + dx][y + dy] == 1 and not b1:
                    b1 = (x + dx, y + dy)
                binimg[x + dx][y + dy] == 1
                if binimg[x + dx][y + dy] == 0 and not c:
                    c = (x + dx, y + dy)


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputimage', type=misc.imread)
    return parser.parse_args()


if __name__ == '__main__':
    main()
