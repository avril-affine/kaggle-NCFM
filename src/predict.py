import os
import sys
import h5py
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from train import BATCH_SIZE, SHEAR, ZOOM, SHIFT, FLIP


HOME_DIR = '/home/panda/Desktop/Projects/fish'

def main():

    weights_dir = sys.argv[1]
    model_dir = os.path.join(weights_dir, 'fold_' + sys.argv[2])
    model_path = os.path.join(model_dir, 'weights.h5')

    test_dir = os.path.join(HOME_DIR, 'data/test')
    test_folder = 'test_stg1'
    num_test = len(os.listdir(os.path.join(test_dir, test_folder)))

    test_gen = ImageDataGenerator(
        rescale=1./255,
        shear_range=SHEAR,
        zoom_range=ZOOM,
        width_shift_range=SHIFT,
        height_shift_range=SHIFT,
        horizontal_flip=FLIP)

    model = load_model(model_path)
    preds = np.zeros((num_test, 8))

    for _ in xrange(10):
        test_generator = test_gen.flow_from_directory(
            test_dir,
            target_size=(299, 299),
            batch_size=BATCH_SIZE,
            shuffle=False)

        preds += model.predict_generator(test_generator, num_test)

    preds /= 10
    results_path = os.path.join(model_dir, 'results.h5')
    with h5py.File(results_path, 'w') as hf:
        hf.create_dataset('results', data=preds)

    with open(os.path.join(HOME_DIR, 'test_filenames.txt'), 'w') as f:
        filenames = [os.path.basename(x) for x in test_generator.filenames]
        f.write('\n'.join(filenames))


if __name__ == '__main__':
    main()
