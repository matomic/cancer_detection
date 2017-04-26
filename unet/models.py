# -*- coding: utf-8 -*-
'''  (^._.^)ﾉ☆( _ _).oO  '''
from __future__ import print_function, division

#from keras import backend as K
from keras.layers import Input
#from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dense,Flatten, Deconvolution2D,Dropout,SpatialDropout2D,Activation,Lambda,AveragePooling2D,GlobalAveragePooling2D,Conv2D, SeparableConv2D,Cropping2D,Convolution3D,MaxPooling3D
#from keras.layers import Merge
#from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import Conv3D as Convolution3D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization as BN
from keras.layers.pooling import MaxPooling3D
from keras.models import Model


def get_3Dnet(name, version, W, H, CH):
    '''  (=^･ｪ･^=))ﾉ彡☆  '''
    inputs = Input((W, H, CH, 1))
    if name == 'Sampe3D':
        if version == 1:
            n0 = 8
            net = Convolution3D(n0, 3, 3, 1, activation='relu', border_mode='same')(inputs)
            net = Convolution3D(n0, 3, 3, 3, activation='relu', border_mode='same')(net)
            net = BN()(net)
            net = MaxPooling3D(pool_size=(2,2,1))(net)

            net = Convolution3D(n0*2, 3, 3, 3, activation='relu', border_mode='same')(net)
            net = Convolution3D(n0*2, 3, 3, 3, activation='relu', border_mode='same')(net)
            net = BN()(net)
            net = MaxPooling3D(pool_size=(2,2,2))(net)

            net = Convolution3D(n0*4, 3, 3, 3, activation='relu', border_mode='same')(net)
            net = Convolution3D(n0*4, 3, 3, 1, activation='relu', border_mode='same')(net)
            net = BN()(net)
            net = MaxPooling3D(pool_size=(2, 2, 2))(net)

            net = Flatten()(net)
            net = Dense(128, activation='relu')(net)
            net = Dropout(0.5)(net)
            labels = Dense(1, activation='sigmoid')(net)
        elif version == 2:
            n0 = 12
            net = Convolution3D(n0, 3, 3, 1, activation='relu', border_mode='same')(inputs)
            net = Convolution3D(n0, 3, 3, 3, activation='relu', border_mode='same')(net)
            net = BN()(net)
            net = MaxPooling3D(pool_size=(2, 2, 1))(net)

            net = Convolution3D(n0, 3, 3, 3, activation='relu', border_mode='same')(net)
            net = Convolution3D(n0, 3, 3, 3, activation='relu', border_mode='same')(net)
            net = BN()(net)
            net = MaxPooling3D(pool_size=(2, 2, 2))(net)

            net = Convolution3D(n0*2, 3, 3, 3, activation='relu', border_mode='same')(net)
            net = Convolution3D(n0*2, 3, 3, 1, activation='relu', border_mode='same')(net)
            net = BN()(net)
            net = MaxPooling3D(pool_size=(2, 2, 2))(net)

            net = Convolution3D(n0*2, 3, 3, 3, activation='relu', border_mode='same')(net)
            net = BN()(net)
            net = MaxPooling3D(pool_size=(2, 2, 2))(net)

            net = Flatten()(net)
            net = Dense(128, activation='relu')(net)
            net = Dropout(0.5)(net)
            labels = Dense(1, activation='sigmoid')(net)
        else:
            raise Exception("not defined net version")

    model = Model(inputs=inputs, outputs=labels)
    return model

# eof
