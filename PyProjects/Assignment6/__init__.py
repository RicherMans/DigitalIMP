#===============================================================================
# In this assignment it is necessary to implement Morphological processing
#===============================================================================
import argparse
from scipy import misc
import cmath
import numpy as np

def main():
    parseArgs()
    


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputimage',type=misc.imread)
    return parser.parse_args()


def decode():
    decodesymbols()
    inversetransform()
    mergesubimages()
def mergesubimages():
    pass
    
def decodesymbols():
    pass
def inversetransform():
    pass
def encode():
    constructsubimages()
    forwardtransform()
    quantitize()
    encodeSymbols()
    
def constructsubimages():
    pass

def forwardtransform():
    pass

def quantitize():
    pass

def encodeSymbols():
    pass

if __name__ == '__main__':
    main()