"""train_classifier.py: Train a classifier keras model."""
import os
import sys
import json
import argparse
import src.utils.base_parser
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from src.utils.run_folds import run_folds
from src.my_iterator import MyDirectoryIterator
from src.my_tensorboard import BatchTensorboard


def main(args):
    # Directories
    HOME_DIR   = args.home_dir
    MODEL_NAME = args.model_name
    data_dir   = args.data_dir
    train_dir  = os.path.join(data_dir, 'train')
    val_dir    = os.path.join(data_dir, 'val')

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
    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    checkpoint  = ModelCheckpoint(os.path.join(weights_dir, 'weights.h5'),
                                  verbose=1,
                                  save_best_only=True)
    tensorboard = BatchTensorboard(log_dir=weights_dir, write_graph=False)
    callbacks   = [checkpoint, tensorboard]

    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

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
        shuffle=True,
        img_info=img_info,
        localizer=localizer,
        num_test=num_test,
        class_mode='categorical')

    val_gen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True)
    val_gen = MyDirectoryIterator(
        val_dir,
        val_gen,
        shuffle=True,
        img_info=img_info,
        localizer=localizer,
        num_test=num_test,
        class_mode='categorical')

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
        'Train a keras model classifier with kfolds or one run.')
    parser = base_parser.train_arguments(parser)

    args = parser.parse_args()
    if args.kfolds:
        run_folds(args, __file__)
    else:
        main(args)
