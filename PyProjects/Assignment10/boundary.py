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


def main():
    follow_boundary((1, 2), test)
# args = parseArgs()
# rescale image to binary
# binimg = np.zeros((args.inputimage.shape))
# binimg[args.inputimage > 1] = 1
# boundary_follow(binimg)


def encode_chain(chain):
	encoding = {()}
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
    # Iterate B is used as a stack to indicate if we should stop iterate or not
    iterateB = [(x, y)]
    # FoundBs is a list of all the coordinates where we found a "1"
    foundbs = []
    # FoundCs is a list of all the coordinates where we found a "0"
    foundcs = []
    b0 = (x, y)
    b1 = None
    while(iterateB):
        bx, by = iterateB.pop()
        for dx, dy in zip(xstep, ystep):
            if array[bx + dx][by + dy] == 1 and (bx + dx, by + dy) not in foundbs:
                foundbs.append((bx + dx, by + dy))
                iterateB.append((bx + dx, by + dy))
            if not b1:
                b1 = (bx + dx, by + dy)
    found = np.zeros(test.shape)
    for corr in foundbs:
        x, y = corr
        found[x, y] = 1
    print found


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
