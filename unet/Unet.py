'''---'''
from keras import backend as K
from keras.layers import Input
from keras.layers.convolutional import Conv2D as Convolution2D, UpSampling2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization as BN
from keras.layers.pooling import MaxPooling2D
from keras.models import Model

def get_unet(img_rows, img_cols, version, **cfg_dict):
    '''Generate {version} UNET model for image of size {img_rows}x{img_cols}'''
    if K.image_dim_ordering() == 'tf':
        c = 3
        inputs = Input((img_rows, img_cols, 1))
    else:
        c = 1
        inputs = Input((1,img_rows, img_cols))

    def unet_conv(x_in, nf, rep=1):
        '''Generate a 2D convolution layer(s) for {x_in} with {nf} filters, repeating {rep} times'''
        # c in bind to local variable of `get_unet`
        x_out = Convolution2D(nf, (3, 3), padding='same', activation='relu')(x_in)
        x_out = BN(axis=c)(x_out)
        #x_out = LeakyReLU(0.1)(x_out)
        if rep>1:
            for _ in range(rep-1):
                x_out = Convolution2D(nf, (3, 3), padding='same', activation='relu')(x_out)
                x_out = BN(axis=c)(x_out)
                #x_out = LeakyReLU(0.1)(x_out)
        return x_out

    if version == 0:
        # dummpy archietecture
        n0 = 8
        #net = unet_conv(inputs, n0, rep=1)
        labels = Convolution2D(1, (3, 3), activation='sigmoid', padding='same', name='labels')(inputs)
    elif version == 1:
        n0 = 8
        net = unet_conv(inputs, n0, rep=2)
        net = MaxPooling2D(pool_size=(2,2))(net)
        net = unet_conv(net, n0*2, rep=2)
        net = MaxPooling2D(pool_size=(2,2))(net)
        net = unet_conv(net, n0*4, rep=2)
        net = MaxPooling2D(pool_size=(2,2))(net)
        net = unet_conv(net, n0*8, rep=2)
        net = MaxPooling2D(pool_size=(2,2))(net)
        net = unet_conv(net, n0*16, rep=2)
        net = UpSampling2D(size=(2,2))(net)
        net = unet_conv(net, n0*8, rep=2)
        net = UpSampling2D(size=(2,2))(net)
        net = unet_conv(net, n0*4, rep=2)
        net = UpSampling2D(size=(2,2))(net)
        net = unet_conv(net, n0*2, rep=2)
        net = UpSampling2D(size=(2,2))(net)
        net = unet_conv(net, n0, rep=1)
        labels = Convolution2D(1, (3, 3), activation='sigmoid', padding='same', name='labels')(net)
    elif version == 2:
        n0 = 8
        net = unet_conv(inputs, n0, rep=3)
        net = MaxPooling2D(pool_size=(2,2))(net)
        cov2 = unet_conv(net, n0*2, rep=2)
        net = MaxPooling2D(pool_size=(2,2))(cov2)
        cov3 = unet_conv(net, n0*4, rep=2)
        net = MaxPooling2D(pool_size=(2,2))(cov3)
        cov4 = unet_conv(net, n0*8, rep=2)
        net = UpSampling2D(size=(2,2))(cov4)
        net = unet_conv(net, n0*4, rep=2)
        net = unet_conv(concatenate([net,cov3], axis=-1), n0*4, rep=1)
        net = UpSampling2D(size=(2,2))(net)
        net = unet_conv(net, n0*2, rep=2)
        net = unet_conv(concatenate([net,cov2], axis=-1), n0*2, rep=1)
        net = UpSampling2D(size=(2,2))(net)
        net = unet_conv(net, n0, rep=1)
        labels = Convolution2D(1, (3, 3), activation='sigmoid', padding='same', name='labels')(net)
    elif version==3:
        sub_rep = cfg_dict['subsampling_conv_repeat'] # number fo time conv2d/BN layers are repeated on the way down
        sup_rep = cfg_dict['upsampling_conv_repeat']  # number of time conv2d/BN layers are repeated on the way up
        n0 = 8                                     #    (?, 512, 512, 1)
        cov1 = unet_conv(inputs, n0, rep=sub_rep)      # -> (?, 512, 512, 8)
        net = MaxPooling2D(pool_size=(2,2))(cov1)  # -> (?, 256, 256, 8)
        cov2 = unet_conv(net, n0*2, rep=sub_rep)       # -> (?, 256, 256, 16)
        net = MaxPooling2D(pool_size=(2,2))(cov2)  # -> (?, 128, 128, 16)
        cov3 = unet_conv(net, n0*4, rep=sub_rep)       # -> (?, 128, 128, 32)
        net = MaxPooling2D(pool_size=(2,2))(cov3)  # -> (?, 64, 64, 32)
        cov4 = unet_conv(net, n0*8, rep=sub_rep)       # -> (?, 64, 64, 64)
        net = MaxPooling2D(pool_size=(2,2))(cov4)  # -> (?, 32, 32, 64)
        cov5 = unet_conv(net, n0*16, rep=sub_rep)      # -> (?, 32, 32, 128)
        #5
        net = UpSampling2D(size=(2,2))(cov5)       # -> (?, 64, 64, 128)
        net = unet_conv(net, n0*8, rep=sup_rep)          # -> (?, 64, 64, 64)
        net = UpSampling2D(size=(2,2))(net)        # -> (?, 128, 128, 64)
        net = unet_conv(net, n0*4, rep=sup_rep)          # -> (?, 128, 128, 32)
        net = UpSampling2D(size=(2,2))(net)        # -> (?, 256, 256, 32)
        net = unet_conv(net, n0*2, rep=sup_rep)          # -> (?, 256, 256, 16)
        net = UpSampling2D(size=(2,2))(net)        # -> (?, 512, 512, 16)
        net5 = unet_conv(net, n0, rep=sup_rep)           # -> (?, 512, 512, 8)
        #4
        net = UpSampling2D(size=(2,2))(cov4)       # -> (?, 128, 128, 64)
        net = unet_conv(net, n0*4, rep=sup_rep)          # -> (?, 128, 128, 32)
        net = UpSampling2D(size=(2,2))(net)        # -> (?, 256, 256, 32)
        net = unet_conv(net, n0*2, rep=sup_rep)          # -> (?, 256, 256, 16)
        net = UpSampling2D(size=(2,2))(net)        # -> (?, 512, 512, 16)
        net4 = unet_conv(net, n0, rep=sup_rep)           # -> (?, 512, 512, 8)
        #3
        net = UpSampling2D(size=(2,2))(cov3)       # -> (?, 256, 256, 32)
        net = unet_conv(net, n0*2, rep=sup_rep)          # -> (?, 256, 256, 16)
        net = UpSampling2D(size=(2,2))(net)        # -> (?, 512, 512, 16)
        net3 = unet_conv(net, n0, rep=sup_rep)           # -> (?, 512, 512, 8)

        net = concatenate([net5,net4,net3], axis=-1) # -> (?, 512, 512, 24)
        labels = Convolution2D(1, (1, 1), activation='sigmoid', padding='same', name='labels')(net) # -> (?, 512, 512, 1)
    else:
        raise Exception("not defined net version")

    model = Model(inputs=inputs, outputs=labels)
    return model
