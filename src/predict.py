import os
import sys
import h5py
import argparse
import numpy as np
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from utils import run_folds


def main(args):
    # Test Augmentation
    N = args.num_test
    SHEAR = args.shear
    ZOOM = args.zoom
    ROTATION = args.rotation
    SHIFT = args.shift
    FLIP_LR = args.flip_lr
    FLIP_UD = args.flip_ud

    # Directories
    HOME_DIR = args.home_dir
    model_dir = args.data_dir
    model_path = os.path.join(model_dir, 'weights.h5')
    output_file = args.output_file

    # Output tensor (only required for multi output models)
    output_tensor_name = args.output_tensor

    test_dir = os.path.join(HOME_DIR, 'data/test')
    test_folder = 'test_stg1'
    num_test = len(os.listdir(os.path.join(test_dir, test_folder)))

    test_gen = ImageDataGenerator(
        rescale=1./255,
        shear_range=SHEAR,
        zoom_range=ZOOM,
        rotation_range=ROTATION,
        width_shift_range=SHIFT,
        height_shift_range=SHIFT,
        horizontal_flip=FLIP)

    model = load_model(model_path)
    if output_tensor:
        for output in model.outputs:
            if output.name == output_tensor_name:
                output_tensor = output
        else:
            raise Exception('Cannot find output tensor for loaded model: '
                            + output_tensor_name)
        model = Model(model.input, output)
    preds = np.zeros((num_test, 8))

    for _ in xrange(N):
        test_generator = test_gen.flow_from_directory(
            test_dir,
            target_size=(299, 299),
            batch_size=BATCH_SIZE,
            shuffle=False)

        preds += model.predict_generator(test_generator, num_test)

    preds /= N
    results_path = os.path.join(model_dir, output_file)
    with h5py.File(results_path, 'w') as hf:
        hf.create_dataset('results', data=preds)

    with open(os.path.join(HOME_DIR, 'test_filenames.txt'), 'w') as f:
        filenames = [os.path.basename(x) for x in test_generator.filenames]
        f.write('\n'.join(filenames))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predict a keras model with kfolds or a single run.')
    parser.add_argument('--home_dir', required=True,
        help='Path to the root directory of the project')
    parser.add_argument('--data_dir', required=True,
        help='Path to a data directory. If kfolds are specified, '
        'the directory should contain a directory for each fold 0 to k-1, '
        'else it should contain a weights.h5 file.')
    parser.add_argument('--output_file', default='results.h5',
        help='Name of output file. This will be written to the data_dir.')
    parser.add_argument('--kfolds', type=int, default=0,
        help='Number of folds to run the model on. The directory structure '
             'must be setup before running this file.')
    parser.add_argument('--fold_prefix', default='fold_',
        help='Prefix for each fold directory.')
    parser.add_argument('--output_tensor', default=None,
        help='Name of the output tensor. Only needed for multi '
             'output models.')
    parser.add_argument('--num_test', type=int, default=1,
        help='Number of tests predictions to average per image.')
    parser.add_argument('--shear', type=float, default=0.,
        help='Data augmentation: shear intensity in radians.')
    parser.add_argument('--zoom', type=float, default=0.,
        help='Data augmentation: range for random zoom.')
    parser.add_argument('--rotation', type=float, default=0.,
        help='Data augmentation: range for random rotation in degrees.')
    parser.add_argument('--shift', type=float, default=0.,
        help='Data augmentation: range for random shifts in x/y directions.')
    parser.add_argument('--flip_lr', action='store_true',
        help='Data augmentation: specify to apply random horizontal flips.')
    parser.add_argument('--flip_ud', action='store_true',
        help='Data augmentation: specify to apply random vertical flips.')
    args = parser.parse_args()

    if args.kfolds:
        run_folds(args, __file__)

        # average results
        for fold in xrange(args.kfolds):
            fold_dir = os.path.join(args.data_dir, args.fold_prefix + str(fold))
            with h5py.File(os.path.join(fold_dir, args.output_file), 'r') as hf:
                data = hf.get('results')[:]
            if fold == 0:
                preds = np.array(data)
            else:
                preds += np.array(data)
        preds /= args.kfolds

        with open('test_filenames.txt', 'r') as f:
            filenames = f.read()
            filenames = filenames.split()

        csv_file, _ = args.output_file.rsplit('.', 1)
        with open(os.path.join(args.data_dir, csv_file + '.csv'), 'w') as f:
            f.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
            for fname, pred in zip(filenames, preds):
                pred = [str(x) for x in pred]
                f.write('{},{}\n'.format(fname, ','.join(pred)))
    else:
        main(args)
