'''
    Specification for a Convolutional Bayesian Neural Net to classify MRI volumes.
    The model takes in an MRI volume and returns the probabilities that it belongs to each of `out_classes` different classes (e.g. 'clean', and 'artefact'). 
'''

from keras.regularizers import L1, L2
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv3D, MaxPooling3D, Input

def getConvNet(out_classes: int = 2, input_shape: tuple = (192, 256, 256, 1)) -> Model:
    """
    Convolutional Bayesian Neural Net classifier architecture.

    :param tuple input_shape: Tuple of image (depth, height, width, num_channels).
    :param int out_classes: Final layer size; if 2, then {first: clean, second: artefact}.

    :returns: Uncompiled keras model.
    :rtype: Model

    .. note::
        The compiled model can also be given a tensor of dimension 
        (batch_size, depth, height, width, num_channels).
    """
    inp = Input(input_shape) # depth x witdth x height x channels

    # 192 x 256 x 256 x 1
    x = Conv3D(8, (3,3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(inp)
    x = BatchNormalization(axis=4)(x)
    x = Activation('relu')(x)

    x = Conv3D(8, (3,3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=4)(x)
    x = Activation('relu')(x)

    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    # 96 x 128 x 128 x 8
    x = Conv3D(16, (3,3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=4)(x)
    x = Activation('relu')(x)

    x = Conv3D(16, (3,3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=4)(x)
    x = Activation('relu')(x)

    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    # 48 x 64 x 64 x 16
    x = Conv3D(32, (3,3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=4)(x)
    x = Activation('relu')(x)

    x = Conv3D(32, (3,3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=4)(x)
    x = Activation('relu')(x)

    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    # 24 x 32 x 32 x 32
    x = Conv3D(64, (3,3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=4)(x)
    x = Activation('relu')(x)

    x = Conv3D(64, (3,3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=4)(x)
    x = Activation('relu')(x)

    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    # 12 x 16 x 16 x 64
    x = Conv3D(128, (3,3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=4)(x)
    x = Activation('relu')(x)

    x = Conv3D(128, (3,3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=4)(x)
    x = Activation('relu')(x)

    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    # 6 x 8 x 8 x 128
    # x = Dropout(0.25)(x,training=True) # = always apply dropout (permanent 'training mode')
    x = Flatten()(x)

    # 49152 x 1
    x = Dense(128, name='dense_pre', kernel_regularizer=L2(l2=0.01))(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    # 128 x 1
    x = Dropout(0.5)(x, training=True) # = always apply dropout (permanent 'training mode')
    out = Dense(out_classes, name='dense_post', activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)

    return model