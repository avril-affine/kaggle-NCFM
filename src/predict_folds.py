import os
import h5py
import numpy as np


HOME_DIR = '/home/panda/Desktop/Projects/fish'


def main():

    weights_dir = os.path.join(
            HOME_DIR, 'models/k_fold_final_low_lr_early_stop')

    for fold in xrange(10):
        fold_dir = os.path.join(weights_dir, 'fold_' + str(fold))
        os.system('python predict.py {} {}'.format(weights_dir, fold))
        with h5py.File(os.path.join(fold_dir, 'results.h5'), 'r') as hf:
            data = hf.get('results')[:]
        if fold == 0:
            preds = np.array(data)
        else:
            preds += np.array(data)

    preds /= 10

    with open('test_filenames.txt', 'r') as f:
        filenames = f.read()
        filenames = filenames.split()

    with open(os.path.join(weights_dir, 'results.csv'), 'w') as f:
        f.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
        for fname, pred in zip(filenames, preds):
            pred = [str(x) for x in pred]
            f.write('{},{}\n'.format(fname, ','.join(pred)))


if __name__ == '__main__':
    main()
