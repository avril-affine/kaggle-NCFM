import os
import sys


def main():
    DATA_DIR = '/home/panda/Desktop/Projects/fish/data/train_folds'
    for fold in xrange(10):
        print '------------------Training Fold %i------------------' % fold
        os.system('python train.py {} {}'.format(DATA_DIR, fold))


if __name__ == '__main__':
    main()
