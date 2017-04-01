import tensorflow as tf
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)
from keras.layers.core import (Dense, Dropout, Flatten, Activation,
                               SpatialDropout2D, Lambda)
from keras.layers import AveragePooling2D, Input
from keras.layers.merge import multiply
from keras.layers.pooling import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.applications.inception_v3 import InceptionV3, conv2d_bn
from keras.models import Model
from keras import backend as K
# from my_keras_model import Model


OUTPUT_NAME = 'output'


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


def localize(fine_tune=False):
    from keras.models import Model
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

    model = Model(model.input, localize)
    return model


def attention_model(fine_tune=False):
    def attention_layer(x):
        attention = Convolution2D(num_filters, (3, 3), padding='same')(x)
        attention = BatchNormalization(axis=3)(attention)
        attention = Activation('sigmoid')(attention)
        attention = K.max(attention, axis=3)
        x = K.permute_dimensions(x, pattern=(0, 3, 2, 1))
        x = tf.multiply(x, attention)
        x = K.permute_dimensions(x, pattern=(0, 3, 2, 1))
        return x

    # 299 x 299 x 3
    num_filters = 64
    input_layer = Input((299, 299, 3))
    x = Convolution2D(num_filters, (3, 3), padding='same')(input_layer)
    x = BatchNormalization(axis=3, scale=True)(x)
    x = Activation('relu')(x)
    x = Convolution2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Lambda(attention_layer)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # 149 x 149 x 64
    num_filters = 128
    x = Convolution2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=3, scale=True)(x)
    x = Activation('relu')(x)
    x = Convolution2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Lambda(attention_layer)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # 74 x 74 x 128
    num_filters = 256
    x = Convolution2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=3, scale=True)(x)
    x = Activation('relu')(x)
    x = Convolution2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Convolution2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Lambda(attention_layer)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # 37 x 37 x 256
    num_filters = 512
    x = Convolution2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=3, scale=True)(x)
    x = Activation('relu')(x)
    x = Convolution2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Convolution2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Lambda(attention_layer)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # 18 x 18 x 512
    num_filters = 512
    x = Convolution2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=3, scale=True)(x)
    x = Activation('relu')(x)
    x = Convolution2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Convolution2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Lambda(attention_layer)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # 9 x 9 x 512
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(8, activation='softmax', name=OUTPUT_NAME)(x)

    model = Model(input_layer, x)
    return model
