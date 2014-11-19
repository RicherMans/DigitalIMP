import argparse
from scipy import misc


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputimage', type=misc.imread)
    return parser.parse_args()


def main():
    args = parseArgs()


if __name__ == '__main__':
    main()