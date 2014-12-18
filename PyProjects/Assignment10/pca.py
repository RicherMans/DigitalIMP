'''
Created on Nov 29, 2014

@author: richman
'''
import argparse
from scipy import misc
import os
import numpy as np
import sklearn

from sklearn import decomposition

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputdir', type=str, help='directory of the pictures')
    return parser.parse_args()


def reconstruct(y,mean_x,mat_A):
    ''' 
    y : an array, which was processed by using PCA, the process will be reversed
    mat_A : a unitary matrix
    mean_x : the mean of the old given data
    '''
    mean_x = mean_x.reshape(6,1)
    ret = np.dot(mat_A.T,y.T) + mean_x
    print ret.shape
    return ret

def pca(arr,n_components):
    #     TODO: No idea why it doesnt work!
    #     Calculate mean for every pixel
    ret =arr.astype(float)
    mean_X = ret.mean(axis=0,dtype=float)
    c_X = 0.
    for vec in ret:
        c_X+= (np.matrix(vec).T* np.matrix(vec)) - np.matrix(mean_X).T * np.matrix(mean_X)
    assert np.array_equal(c_X,c_X.T) 
    _,V =  np.linalg.eigh(c_X)
#     Reverse the order, to be ascending
    revV = V[::-1]
#     Check if the resulting matrix is hermitian
    assert np.allclose(revV.T,np.linalg.inv(revV))
#     Generate the matrix A, which is defined as having its rows set as the eienvectors of c_x
    A = revV[0:n_components]
    
    samples, _ = arr.shape
    y = np.empty((samples,n_components))
    for i in range(len(arr)):
        y[i] = np.dot(A,arr[i]-mean_X)
    return y,mean_X,A

def main():
    args = parseArgs()
    picpaths = [os.path.join(os.getcwd(), args.inputdir, x)
                for x in os.listdir(args.inputdir)]
    inputimages = [misc.imread(pic).flatten() for pic in picpaths]
    num_imgs =  len(inputimages)

#     First of all we transform the input from (6,N,N) to (N,N,6), getting
#   for every pixel n,n a 6 dimensional vector.
    # create matrix to store all flattened images
    imgs = np.array(inputimages).T
    n_components = 2
    proj,mean_x,transformMat = pca(imgs,n_components)
    reconstructedimgs = reconstruct(proj, mean_x, transformMat)
    proj = proj.reshape((564,564,n_components))
    for pc_component,pcimg in zip(range(1,n_components+1),proj.T):
        misc.imsave('PC_Component_%i.tif'%pc_component,pcimg.T )
    reconstructedimgs = reconstructedimgs.reshape(6,564,564)
    for pc_component,pcimg in zip(range(1,num_imgs+1),reconstructedimgs.T):
        misc.imsave('Reconstructed_%i.tif'%(pc_component),pcimg.T)
        

if __name__ == '__main__':
    main()


