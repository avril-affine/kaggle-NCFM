import os
import sys
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Flatten, Dense, AveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from my_iterator import MyDirectoryIterator
from models import classify, localize_classify
from my_tensorboard import BatchTensorboard


HOME_DIR = '/home/panda/Desktop/Projects/fish'

LR = 0.000001
EPOCHS = 100
BATCH_SIZE = 32

SHEAR = 0.1
ZOOM = 0.1
ROTATION = 10.
SHIFT = 0.1
FLIP = True

MULTI_OUTPUT = True


def main():
    data_dir = sys.argv[1]
    fold = sys.argv[2]
    if MULTI_OUTPUT:
        box_file = sys.argv[3]
    train_dir = os.path.join(data_dir, 'fold_' + fold, 'train')
    val_dir = os.path.join(data_dir, 'fold_' + fold,  'val')
    weights_dir = os.path.join(
        HOME_DIR, 'models/k_fold_low_final_low_lr_early_stop/fold_' + fold)

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

    model = localize_classify()

    tensorboard = BatchTensorboard(log_dir=weights_dir, write_graph=False)
    early_stop = EarlyStopping(patience=7, verbose=1)

    adam = Adam(lr=LR)
    if MULTI_OUTPUT:
        model.compile(optimizer=adam,
                      loss={'localize': 'mse',
                            'classify': 'categorical_crossentropy'},
                      metrics={'classify': 'accuracy'},
                      loss_weights={'localize': 0.05,
                                    'classify': 1.})
    else:
        model.compile(optimizer=adam, loss='categorical_crossentropy',
                      metrics=['accuracy'])

    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    checkpoint = ModelCheckpoint(os.path.join(weights_dir, 'weights.h5'),
                                 verbose=1,
                                 save_best_only=True)

    train_gen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=SHEAR,
        zoom_range=ZOOM,
        rotation_range=ROTATION,
        width_shift_range=SHIFT,
        height_shift_range=SHIFT,
        horizontal_flip=True)
    if MULTI_OUTPUT:
        train_gen = MyDirectoryIterator(
            box_file,
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
            box_file,
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
        callbacks=[checkpoint, tensorboard, early_stop])


if __name__ == '__main__':
    main()
