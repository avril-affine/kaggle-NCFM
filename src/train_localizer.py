import os
import sys
import json
import argparse
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from utils.run_folds import run_folds
from my_iterator import MyDirectoryIterator
from my_tensorboard import BatchTensorboard
from models import OUTPUT_NAME


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
    parser = argparse.ArgumentParser(
        description='Train localizer a keras model with kfolds or a one run.')
    parser.add_argument(
        '--home_dir', required=True,
        help='Path to the root directory of the project')
    parser.add_argument(
        '--data_dir', required=True,
        help='Path to a data directory. If kfolds are specified, '
        'the directory should contain a directory for each fold 0 to k-1, '
        'else it should contain a train and val folder.')
    parser.add_argument(
        '--model_name', required=True,
        help='Name of the model. This will also be the name of the directory '
             'that stores the model weights and run data.')
    parser.add_argument(
        '--import_model', required=True,
        help='Model to import and run in the form of '
             'path.to.script.function: where `path.to.script` is the module '
             'to import and `function` is the function within the module.')
    parser.add_argument(
        '--bbox_file', required=True,
        help='Path to a json file containing a map from img basename to '
             'bounding box.')
    parser.add_argument(
        '--learning_rate', type=float, default=0.0001,
        help='Learning rate of the model.')
    parser.add_argument(
        '--epochs', type=int, default=10,
        help='Number of epochs to run the model.')
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for the model.')
    parser.add_argument(
        '--shear', type=float, default=0.,
        help='Data augmentation: shear intensity in radians.')
    parser.add_argument(
        '--zoom', type=float, default=0.,
        help='Data augmentation: range for random zoom.')
    parser.add_argument(
        '--rotation', type=float, default=0.,
        help='Data augmentation: range for random rotation in degrees.')
    parser.add_argument(
        '--shift', type=float, default=0.,
        help='Data augmentation: range for random shifts in x/y directions.')
    parser.add_argument(
        '--fine_tune', action='store_true',
        help='Specify to freeze the pretrained weights.')
    parser.add_argument(
        '--flip_lr', action='store_true',
        help='Data augmentation: specify to apply random horizontal flips.')
    parser.add_argument(
        '--flip_ud', action='store_true',
        help='Data augmentation: specify to apply random vertical flips.')
    args = parser.parse_args()

    if args.kfolds:
        run_folds(args, __file__)
    else:
        main(args)
