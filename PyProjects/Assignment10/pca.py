'''
Created on Nov 29, 2014

@author: richman
'''
import argparse
from scipy import misc
import os
import numpy as np

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputdir', type=str, help='directory of the pictures')
    parser.add_argument('-n', type=int, help='Number of components which should be kept',default=2)
    return parser.parse_args()


def reconstruct(y, mean_x, mat_A):
    ''' 
    y : an array, which was processed by using PCA, the process will be reversed
    mat_A : a unitary matrix
    mean_x : the mean of the old given data
    '''
    mean_x = mean_x.reshape(6, 1)
    ret = np.dot(mat_A.T, y) + mean_x
    return ret

def pca(arr, n_components):
    #     TODO: No idea why it doesnt work!
    #     Calculate mean for every pixel
    ret = arr.astype(float)
    mean_X = ret.mean(axis=1, dtype=float, keepdims=True)
    c_X = np.cov(arr)
    x, V = np.linalg.eigh(c_X)
#     Reverse the order, to be ascending
    revV = V[::-1]
#     Check if the resulting matrix is hermitian
    assert np.allclose(revV.T, np.linalg.inv(revV))
#     Generate the matrix A, which is defined as having its rows set as the eienvectors of c_x
    A = revV[0:n_components]
    
    np.set_printoptions(linewidth=150)
    
    y = np.dot(A, (arr - mean_X))
    y= scale(y)
    return y, mean_X, A

def scale(arr):
    min_val = np.min(arr)
    scaled = arr - min_val
    scaled[scaled <= np.finfo(np.float32).eps] = 0.
    max_value = np.max(scaled)
    if max_value != 0. : scaled = scaled * (255. / max_value)
    return scaled

def main():
    args = parseArgs()
    picpaths = [os.path.join(os.getcwd(), args.inputdir, x)
                for x in os.listdir(args.inputdir)]
    inputimages = [misc.imread(pic).flatten() for pic in picpaths]
    num_imgs = len(inputimages)

#     First of all we transform the input from (6,N,N) to (N,N,6), getting
#   for every pixel n,n a 6 dimensional vector.
    # create matrix to store all flattened images
    imgs = np.array(inputimages)
    n_components = args.n
    proj, mean_x, transformMat = pca(imgs, n_components)
    reconstructedimgs = reconstruct(proj, mean_x, transformMat)
    proj = proj.reshape((n_components, 564, 564))
    for pc_component, pcimg in zip(range(1, n_components + 1), proj):
        misc.imsave('PC_Component_%i.tif' % pc_component, pcimg)
    reconstructedimgs = reconstructedimgs.reshape(6, 564, 564)
    for pc_component, pcimg in zip(range(1, num_imgs + 1), reconstructedimgs):
        misc.imsave('Reconstructed_%i.tif' % (pc_component), pcimg)
    for i in range(num_imgs):
        reconst = imgs[i].reshape(564, 564) - reconstructedimgs[i]
        scale(reconst)
        misc.imsave('Diff_%i.tif' % (i + 1), reconst)
        

if __name__ == '__main__':
    main()


