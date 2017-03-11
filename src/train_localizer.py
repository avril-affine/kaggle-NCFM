"""train_localizer.py: Train a localize model using keras."""
import os
import sys
import json
import argparse
import src.utils.base_parser
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from utils.run_folds import run_folds
from src.my_iterator import MyDirectoryIterator
from src.my_tensorboard import BatchTensorboard


def main(args):
    # Directories
    HOME_DIR   = args.home_dir
    MODEL_NAME = args.model_name
    data_dir   = args.data_dir
    train_dir  = os.path.join(data_dir, 'train')
    val_dir    = os.path.join(data_dir, 'val')

    fold = os.path.basename(data_dir.rstrip('/'))
    if fold.startswith(args.fold_prefix):   # is running kfolds
        weights_dir = os.path.join(HOME_DIR, 'models', MODEL_NAME, fold)
    else:
        weights_dir = os.path.join(HOME_DIR, 'models', MODEL_NAME)

    # Model parameters
    module, function = args.import_model.rsplit('.', 1)
    import_model = __import__(module, fromlist=[function]).__dict__[function]
    LR           = args.learning_rate
    EPOCHS       = args.epochs
    BATCH_SIZE   = args.batch_size
    FINE_TUNE    = args.fine_tune

    # Data augmentation parameters
    SHEAR    = args.shear
    ZOOM     = args.zoom
    ROTATION = args.rotation
    SHIFT    = args.shift
    FLIP_LR  = args.flip_lr
    FLIP_UD  = args.flip_ud

    # Bounding box dict
    with open(args.bbox_file, 'r') as f:
        bbox_dict = json.load(f)

    # Calculate train/val size
    train_size = 0
    val_size   = 0
    train_folders = os.listdir(train_dir)
    for folder in train_folders:
        if folder.startswith('.'):
            continue
        train_folder_path = os.path.join(train_dir, folder)
        val_folder_path   = os.path.join(val_dir, folder)
        train_size += len(os.listdir(train_folder_path))
        val_size   += len(os.listdir(val_folder_path))

    # Model
    model = import_model(FINE_TUNE)
    adam  = Adam(lr=LR)
    model.compile(optimizer=adam, loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    # Callbacks
    checkpoint  = ModelCheckpoint(os.path.join(weights_dir, 'weights.h5'),
                                  verbose=1,
                                  save_best_only=True)
    tensorboard = BatchTensorboard(log_dir=weights_dir, write_graph=False)
    callbacks   = [checkpoint, tensorboard]

    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    # Train/Val Image Generators
    train_gen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        fill_mode='constant',
        shear_range=SHEAR,
        zoom_range=ZOOM,
        rotation_range=ROTATION,
        width_shift_range=SHIFT,
        height_shift_range=SHIFT,
        horizontal_flip=FLIP_LR,
        vertical_flip=FLIP_UD)
    train_gen = MyDirectoryIterator(
        train_dir,
        train_gen,
        target_size=(299, 299),
        batch_size=BATCH_SIZE,
        shuffle=True,
        class_mode='bounding_box',
        bbox_dict=bbox_dict)
    val_gen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True)
    train_gen = MyDirectoryIterator(
        val_dir,
        val_gen,
        target_size=(299, 299),
        batch_size=BATCH_SIZE,
        shuffle=True,
        class_mode='bounding_box',
        bbox_dict=bbox_dict)

    model.fit_generator(
        train_gen,
        samples_per_epoch=train_size,
        nb_epoch=EPOCHS,
        validation_data=val_gen,
        nb_val_samples=val_size,
        verbose=1,
        callbacks=callbacks)


if __name__ == '__main__':
    parser = base_parser.base_parser(
        'Train a keras model localizer with kfolds or one run.')
    parser = base_parser.train_arguments(parser)
    parser.add_argument(
        '--bbox_file', required=True,
        help='Path to a json file containing a map from img basename to '
             'bounding box.')

    args = parser.parse_args()
    if args.kfolds:
        run_folds(args, __file__)
    else:
        main(args)
