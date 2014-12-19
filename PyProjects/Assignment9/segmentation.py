'''
Created on Nov 28, 2014

@author: richman
'''

import argparse
from scipy import misc
import numpy as np

GRAYLEVEL = 256


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputimage', type=misc.imread)
    return parser.parse_args()


def main():
    args = parseArgs()
    sepOtsuimg = separateOtsu(args.inputimage)
    misc.imsave('Otsu.tif', sepOtsuimg)
    sepglobal = separateGlobal(args.inputimage)
    misc.imsave('global.tif', sepglobal)


def separateGlobal(picture):
    threshold_before = np.mean(picture)
    
    threshold_after = threshold_before+20
    g_1 = None
    g_2 = None
    while abs(threshold_after - threshold_before) > 0.01:
        threshold_before = threshold_after
        g_1 = picture[picture > threshold_before]
        g_2 = picture[picture <= threshold_before]
        m1 = np.mean(g_1)
        m2 = np.mean(g_2)
        threshold_after = 0.5 * (m1 + m2)
    output = np.zeros((picture.shape),np.uint8)
    output[picture > threshold_after] = 255
    return output


def separateOtsu(picture):
    hist_count = np.array([0. for _ in range(GRAYLEVEL)])
    for x in range(len(picture)):
        for y in range(len(picture[0])):
            grayscale = picture[x][y]
            hist_count[grayscale] += 1
    sumHist = float(sum(hist_count))
    # Compute average dist
    dist = [float(graylvl) / sumHist for graylvl in hist_count]
    #     Compute culdist
    cul_dist = [sum(dist[:i+1]) for i in range(len(dist))]
    #     Compute Cul Mean dist
    mean_cul_dist=[]
    for i in range(len(dist)):
        curdist = 0.
        for p in range(i+1):
            curdist += p * dist[p]
        mean_cul_dist.append(curdist)
    #     index_hist = [ int(((GRAYLEVEL - 1) * cul_dist[i])) for i in range(histlength)]
    global_mean = sum([i * dist[i] for i in range(len(dist))])
    #      Between class variance [m_G * P_l(k) - m(k)]^2/P_l(k)[1-P_l(k)]
    between_class_var = np.array([(global_mean * cul_dist[k] - mean_cul_dist[k]) ** 2. / (cul_dist[
                                 k] * (1 - cul_dist[k])) if cul_dist[k] != 0 else 1 for k in range(len(dist))])
    
    
    #     Compute the globalvariance
    global_var = sum([(i - global_mean) ** 2 * dist[i]
                      for i in range(len(dist))])
    # Compute K_star
    k_star = between_class_var.argmax(axis=0)

    max_sep = between_class_var[k_star] / global_var
    ret = np.zeros((picture.shape))
    ret[picture > k_star] = 1
    return ret


if __name__ == '__main__':
    main()
