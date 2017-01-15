import os
import sys
import argparse
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from utils.run_folds import run_folds
from my_iterator import MyDirectoryIterator
from my_tensorboard import BatchTensorboard


def main(args):
    # Model parameters
    module, function = args.import_model.rsplit('.', 1)
    import_model = __import__(module, fromlist=[function]).__dict__[function]
    LR = args.learning_rate
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    FINE_TUNE = args.fine_tune
    MULTI_OUTPUT = args.multi_output
    if MULTI_OUTPUT and not args.multi_labels:
        print '--multi_labels is required if MULTI_OUTPUT is specified'
        sys.exit(1)
    MULTI_LABEL_FILE = args.multi_labels
    EARLY_STOP = args.early_stop

    # Data augmentation parameters
    SHEAR = args.shear
    ZOOM = args.zoom
    ROTATION = args.rotation
    SHIFT = args.shift
    FLIP_LR = args.flip_lr
    FLIP_UD = args.flip_ud

    # Directories
    HOME_DIR = args.home_dir
    MODEL_NAME = args.model_name
    data_dir = args.data_dir
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    if data_dir.endswith('/'):
        fold = os.path.basename(data_dir[:-1])
    else:
        fold = os.path.basename(data_dir)
    if fold.startswith(args.fold_prefix):   # is running kfolds
        weights_dir = os.path.join(HOME_DIR, 'models', MODEL_NAME, fold)
    else:
        weights_dir = os.path.join(HOME_DIR, 'models', MODEL_NAME)

    train_folders = os.listdir(train_dir)
    train_size = 0
    val_size = 0
    for folder in train_folders:
        if folder.startswith('.'):
            continue
        train_folder_path = os.path.join(train_dir, folder)
        val_folder_path = os.path.join(val_dir, folder)
        train_size += len(os.listdir(train_folder_path))
        val_size += len(os.listdir(val_folder_path))

    model = import_model(FINE_TUNE)

    adam = Adam(lr=LR)
    if MULTI_OUTPUT:
        model.compile(optimizer=adam,
                      loss={'localize': 'mse',
                            'classify': 'categorical_crossentropy'},
                      metrics={'classify': 'accuracy'},
                      loss_weights={'localize': args.multi_weight,
                                    'classify': 1.})
    else:
        model.compile(optimizer=adam, loss='categorical_crossentropy',
                      metrics=['accuracy'])

    # Callbacks
    checkpoint = ModelCheckpoint(os.path.join(weights_dir, 'weights.h5'),
                                 verbose=1,
                                 save_best_only=True)
    tensorboard = BatchTensorboard(log_dir=weights_dir, write_graph=False)
    callbacks = [checkpoint, tensorboard]
    if EARLY_STOP:
        early_stop = EarlyStopping(patience=7, verbose=1)
        callbacks += [early_stop]

    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    train_gen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=SHEAR,
        zoom_range=ZOOM,
        rotation_range=ROTATION,
        width_shift_range=SHIFT,
        height_shift_range=SHIFT,
        horizontal_flip=FLIP_LR,
        vertical_flip=FLIP_UD)
    if MULTI_OUTPUT:
        train_gen = MyDirectoryIterator(
            MULTI_LABEL_FILE,
            train_dir,
            train_gen,
            target_size=(299, 299),
            batch_size=BATCH_SIZE,
            shuffle=True,
            class_mode='categorical')
    else:
        train_gen = train_gen.flow_from_directory(
            train_dir,
            target_size=(299, 299),
            batch_size=BATCH_SIZE,
            shuffle=True,
            class_mode='categorical')

    val_gen = ImageDataGenerator(rescale=1. / 255)
    if MULTI_OUTPUT:
        val_gen = MyDirectoryIterator(
            MULTI_LABEL_FILE,
            val_dir,
            val_gen,
            target_size=(299, 299),
            batch_size=BATCH_SIZE,
            shuffle=True,
            class_mode='categorical')
    else:
        val_gen = val_gen.flow_from_directory(
            val_dir,
            target_size=(299, 299),
            batch_size=BATCH_SIZE,
            shuffle=True,
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
    parser = argparse.ArgumentParser(
        description='Train a keras model with kfolds or a single run.')
    parser.add_argument('--home_dir', required=True,
        help='Path to the root directory of the project')
    parser.add_argument('--data_dir', required=True,
        help='Path to a data directory. If kfolds are specified, '
        'the directory should contain a directory for each fold 0 to k-1, '
        'else it should contain a train and val folder.')
    parser.add_argument('--model_name', required=True,
        help='Name of the model. This will also be the name of the directory '
             'that stores the model weights and run data.')
    parser.add_argument('--import_model', required=True,
        help='Model to import and run in the form of '
             'path.to.script.function: where `path.to.script` is the module '
             'to import and `function` is the function within the module.')
    parser.add_argument('--kfolds', type=int, default=0,
        help='Number of folds to run the model on. The directory structure '
             'must be setup before running this file.')
    parser.add_argument('--fold_prefix', default='fold_',
        help='Prefix for each fold directory.')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
        help='Learning rate of the model.')
    parser.add_argument('--epochs', type=int, default=10,
        help='Number of epochs to run the model.')
    parser.add_argument('--batch_size', type=int, default=32,
        help='Batch size for the model.')
    parser.add_argument('--fine_tune', action='store_true',
        help='Specify to freeze the pretrained weights.')
    parser.add_argument('--early_stop', type=int, default=0,
        help='Number of epochs without improvement to wait before stopping. '
             'Default to run all epochs.')
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
    parser.add_argument('--multi_output', action='store_true',
        help='Give the model multiple outputs, currently bounding boxes.')
    parser.add_argument('--multi_labels', default=None,
        help='Required if multi_output is set. Path to a json file containing '
             'maps from img to second label, currently bounding boxes.')
    parser.add_argument('--multi_weight', type=float, default=1.,
        help='Required if multi_output is set. Weight for the second output '
             'relative to the logloss output.')
    args = parser.parse_args()

    if args.kfolds:
        run_folds(args, __file__)
    else:
        main(args)
