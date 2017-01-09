from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)
from keras.layers.core import (Dense, Dropout, Flatten, Activation,
                               SpatialDropout2D)
from keras.layers import AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import Sequential
from keras.applications.inception_v3 import InceptionV3
from my_keras_model import My_Model as Model


def localizer(dropout=0., conv_l2=0.0005, fc_l2=0.01):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 256, 256)))
    model.add(Convolution2D(16, 3, 3, W_regularizer=l2(conv_l2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))
    if dropout > 0:
        model.add(SpatialDropout2D(dropout))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, W_regularizer=l2(conv_l2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))
    if dropout > 0:
        model.add(SpatialDropout2D(dropout))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, W_regularizer=l2(conv_l2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))
    if dropout > 0:
        model.add(SpatialDropout2D(dropout))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, W_regularizer=l2(conv_l2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))
    if dropout > 0:
        model.add(SpatialDropout2D(dropout))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, W_regularizer=l2(conv_l2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))
    if dropout > 0:
        model.add(SpatialDropout2D(dropout))

    model.add(Flatten())
    model.add(Dense(4, W_regularizer=l2(fc_l2)))

    return model


def classify(fine_tune=False):
    model = InceptionV3(include_top=False, input_shape=(299, 299, 3))
    if fine_tune:
        for layer in model.layers:
            layer.trainable = False
    output = model.get_layer(index=-1).output
    output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
    output = Flatten(name='flatten')(output)
    output = Dense(8, activation='softmax', name='output')(output)

    model = Model(model.input, output)
    return model


def localize_classify(fine_tune=False):
    model = InceptionV3(include_top=False, input_shape=(299, 299, 3))
    if fine_tune:
        for layer in model.layers:
            layer.trainable = False
    output = model.get_layer(index=-1).output
    localize = Convolution2D(4, 1, 1)(output)
    localize = GlobalAveragePooling2D(name='localize')(localize)
    classify = Convolution2D(8, 1, 1)(output)
    classify = GlobalAveragePooling2D(name='classify')(classify)

    model = Model(model.input, [localize, classify])
    return model
