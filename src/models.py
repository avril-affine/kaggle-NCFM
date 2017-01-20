from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)
from keras.layers.core import (Dense, Dropout, Flatten, Activation,
                               SpatialDropout2D)
from keras.layers import AveragePooling2D, merge
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import Sequential
from keras.applications.inception_v3 import InceptionV3, conv2d_bn
from my_keras_model import Model


OUTPUT_NAME = 'output'


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
    model.add(Dense(4, W_regularizer=l2(fc_l2)), name=OUTPUT_NAME)

    return model


def classify(fine_tune=False):
    model = InceptionV3(include_top=False, input_shape=(299, 299, 3))
    if fine_tune:
        for layer in model.layers:
            layer.trainable = False
    output = model.get_layer(index=-1).output
    output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
    output = Flatten(name='flatten')(output)
    output = Dense(8, activation='softmax', name=OUTPUT_NAME)(output)

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
    classify = GlobalAveragePooling2D(name=OUTPUT_NAME)(classify)
    classify = Activation('softmax', name=OUTPUT_NAME)(classify)

    model = Model(model.input, [localize, classify])
    return model


def localize_classify_deep(fine_tune=False):
    model = InceptionV3(include_top=False, input_shape=(299, 299, 3))
    if fine_tune:
        for layer in model.layers:
            layer.trainable = False
    output = model.get_layer(index=-1).output
    # 8x8x1024
    branch1x1 = conv2d_bn(output, 160, 1, 1)

    branch3x3 = conv2d_bn(output, 192, 1, 1)
    branch3x3_1 = conv2d_bn(branch3x3, 192, 1, 3)
    branch3x3_2 = conv2d_bn(branch3x3, 192, 3, 1)
    branch3x3 = merge([branch3x3_1, branch3x3_2],
                      mode='concat', concat_axis=-1)

    branch3x3dbl = conv2d_bn(output, 224, 1, 1)
    branch3x3dbl = conv2d_bn(output, 192, 3, 3)
    branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 192, 1, 3)
    branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 192, 3, 1)
    branch3x3dbl = merge([branch3x3dbl_1, branch3x3dbl_2],
                         mode='concat', concat_axis=-1)

    branch_pool = AveragePooling2D(
        (3, 3), strides=(1, 1), border_mode='same')(output)
    branch_pool = conv2d_bn(branch_pool, 96, 1, 1)
    output = merge([branch1x1, branch3x3, branch3x3dbl, branch_pool],
                   mode='concat', concat_axis=-1)

    # 8x8x512
    branch1x1 = conv2d_bn(output, 80, 1, 1)

    branch3x3 = conv2d_bn(output, 96, 1, 1)
    branch3x3_1 = conv2d_bn(branch3x3, 96, 1, 3)
    branch3x3_2 = conv2d_bn(branch3x3, 96, 3, 1)
    branch3x3 = merge([branch3x3_1, branch3x3_2],
                      mode='concat', concat_axis=-1)

    branch3x3dbl = conv2d_bn(output, 112, 1, 1)
    branch3x3dbl = conv2d_bn(output, 96, 3, 3)
    branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 96, 1, 3)
    branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 96, 3, 1)
    branch3x3dbl = merge([branch3x3dbl_1, branch3x3dbl_2],
                         mode='concat', concat_axis=-1)

    branch_pool = AveragePooling2D(
        (3, 3), strides=(1, 1), border_mode='same')(output)
    branch_pool = conv2d_bn(branch_pool, 48, 1, 1)
    output = merge([branch1x1, branch3x3, branch3x3dbl, branch_pool],
                   mode='concat', concat_axis=-1)

    localize = Convolution2D(4, 1, 1)(output)
    localize = GlobalAveragePooling2D(name='localize')(localize)
    classify = Convolution2D(8, 1, 1)(output)
    classify = GlobalAveragePooling2D()(classify)
    classify = Activation('softmax', name=OUTPUT_NAME)(classify)

    model = Model(model.input, [localize, classify])
    return model
