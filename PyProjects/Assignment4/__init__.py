from argparse import ArgumentParser
from scipy import misc
import cmath
import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform
from numpy import real, matlib

def main():
    args = parseargs()
    imagearr = misc.imread(args.inputimage)
    
    
    noisedimg = imagearr + gaussiannoise(1,1 , imagearr)
    
def gaussiannoise(mean,var,data):
    return np.random.multivariate_normal(mean/255,var/255,data.shape)

def parseargs():
    parser = ArgumentParser()
    parser.add_argument('inputimage', help='The inputted image')
    parser.add_argument('-o', type=str, help='Output image path')
    return parser.parse_args()
    

if __name__ == '__main__':
    main() 