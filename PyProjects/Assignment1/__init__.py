from scipy import misc
from argparse import ArgumentParser
import numpy as np
from matplotlib import pyplot 
from math import floor


GRAYLEVEL = 256

def main():
    args = parseargs()
    picture = misc.imread(args.inputimage)
    transformedpic = transform(picture)
    if args.printPicture:
        misc.imsave(args.printPicture, transformedpic)
    if args.printHistOrg:
        plotHistogram(picture,args.printHistOrg)
    if args.printHistTrans:
        plotHistogram(transformedpic,args.printHistTrans)


def plotHistogram(hist_count,path):
    histogram, bins = np.histogram(hist_count, bins=GRAYLEVEL)
    center = (bins[:-1] + bins[1:]) / 2
    width = 0.50
    ax = pyplot.subplot()
    ax.set_ylabel('Counts')
    ax.set_xlabel('Graylevel')
    ax.set_xticks(np.arange(0,GRAYLEVEL,30))
    ax.set_title(('Histogram'))
    ax.bar(center, histogram, width=width, color='y')
    pyplot.savefig(path)
    pyplot.clf()
    
def equalize(histogram):
    sumHist = float(sum(histogram))
    histlength = len(histogram)
    dist = [ float(graylvl) / sumHist for graylvl in histogram]
    cul_dist = [ sum(dist[:i + 1]) for i in range(len(dist))]
    index_hist = [ int(((GRAYLEVEL - 1) * cul_dist[i])) for i in range(histlength)]
    return index_hist

def genHistogram(picture):
    hist_count = np.array([i for i in range(GRAYLEVEL)])
    for x in range(len(picture)):
        for y in range(len(picture[0])):
            grayscale = picture[x][y]
            hist_count[grayscale] += 1
    return hist_count

def transform(picture):
    histogram_counts = genHistogram(picture)
    equalize_indexes = equalize(histogram_counts)
    transformedpicture = np.zeros(picture.shape)
    for x in range(len(picture)):
        for y in range(len(picture[0])):
            transformedpicture[x][y] = equalize_indexes[picture[x][y]]
    return transformedpicture

def parseargs():
    parser = ArgumentParser()
    parser.add_argument('inputimage', type=file)
    parser.add_argument('-oh1','--printHistOrg',type=str,help='Writes the original histogram out to path ')
    parser.add_argument('-oh2','--printHistTrans',type=str,help='Writes the transformed histogram out to path')
    parser.add_argument('-o','--printPicture',type=str,help='Writes out the transformed Picture')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
