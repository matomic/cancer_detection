# -*- coding: utf-8 -*-
'''  (^._.^)ﾉ☆( _ _).oO  '''
from __future__ import print_function, division
import functools

from keras import backend as K

#from keras import backend as K
from keras.layers import Input
#from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dense,Flatten, Deconvolution2D,Dropout,SpatialDropout2D,Activation,Lambda,AveragePooling2D,GlobalAveragePooling2D,Conv2D, SeparableConv2D,Cropping2D,Convolution3D,MaxPooling3D
#from keras.layers import Merge
#from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import Conv3D as Convolution3D, Conv2D as Convolution2D, UpSampling2D, UpSampling3D
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers.merge import concatenate, add, average
from keras.layers.normalization import BatchNormalization as BN
from keras.layers.pooling import MaxPooling2D, MaxPooling3D, AveragePooling3D

from keras.models import Model


def get_net(netspec):
	'''Return appropriate network specified by {netspec}'''
	if netspec.name.upper() == 'SAMPLE3D':
		return get_3Dnet(netspec.version, netspec.WIDTH, netspec.HEIGHT, netspec.CHANNEL)
	elif netspec.name.upper() == 'UNET':
		return get_unet(netspec.version, netspec.WIDTH, netspec.HEIGHT, **netspec.params)
	else:
		raise ValueError("Unknown network {}: {}".format(netspec.name, netspec))

def get_3Dnet(version, W, H, CH):
	'''fetch 3D network by {name} and {version} for input of shape ({W}, {H}, {CH})'''
	inputs = Input((W, H, CH, 1))

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

def unet_conv(x_in, nf, axis, rep=1, res_like=False):
	'''Generate a 2D convolution layer(s) for {x_in} with {nf} filters, repeating {rep} times'''
	for _ in range(rep):
		conv = Convolution2D(nf, (3, 3), padding='same', activation='relu')(x_in)
		if res_like:
			# resNet-like
			x_in_shape = x_in.shape
			conv_shape = conv.shape
			mul = conv_shape[-1].value / x_in_shape[-1].value
			if mul == 1:
				conv = add([conv, x_in])
			elif int(mul) == mul:
				x_in = Reshape(x_in_shape.as_list()[1:]+[1])(x_in)
				x_in = UpSampling3D((1,1,int(mul)))(x_in) # stupid Keras hackery...
				x_in = Reshape(conv_shape.as_list()[1:])(x_in)
				conv = add([conv, x_in])
			else:
				mul = 1 / mul
				if int(mul) == mul:
					x_in = Reshape(x_in_shape.as_list()[1:]+[1])(x_in)
					x_in = AveragePooling3D((1,1,int(mul)))(x_in) # stupid hackery...
					x_in = Reshape(conv_shape.as_list()[1:])(x_in)
					conv = add([conv, x_in])
				else:
					assert False, 'cannot join shapes {} -> {}'.format(x_in.shape, conv.shape)
		x_in = BN(axis=axis)(conv)
	return x_in
	## c in bind to local variable of `get_unet`
	#x_out = Convolution2D(nf, (3, 3), padding='same', activation='relu')(x_in)
	#x_out = BN(axis=axis)(x_out)
	##x_out = LeakyReLU(0.1)(x_out)
	#if rep>1:
	#	for _ in range(rep-1):
	#		x_out = Convolution2D(nf, (3, 3), padding='same', activation='relu')(x_out)
	#		x_out = BN(axis=axis)(x_out)
	#		#x_out = LeakyReLU(0.1)(x_out)
	#		return x_out

def get_unet(version, W, H, **cfg_dict):
	'''Generate {version} UNET model for image of size {W}x{H}'''
	if K.image_dim_ordering() == 'tf':
		c = 3
		inputs = Input((W, H, 1))
	else:
		c = 1
		inputs = Input((1, W, H))

	def unet_conv_down(x_in, nf):
		'''Generate downsampling convolution layer'''
		x_out = Convolution2D(nf, (3, 3), strides=2, padding='same', activation='relu')(x_in)
		x_out = BN(axis=c)(x_out)
		return x_out

	if version == 0:
		# dummpy archietecture
		n0 = 8
		#net = unet_conv(inputs, n0, rep=1)
		labels = Convolution2D(1, (3, 3), activation='sigmoid', padding='same', name='labels')(inputs)
	elif version == 1:
		n0 = 8
		net = unet_conv(inputs, n0, axis=c, rep=2)
		net = MaxPooling2D(pool_size=(2,2))(net)
		net = unet_conv(net, n0*2, axis=c, rep=2)
		net = MaxPooling2D(pool_size=(2,2))(net)
		net = unet_conv(net, n0*4, axis=c, rep=2)
		net = MaxPooling2D(pool_size=(2,2))(net)
		net = unet_conv(net, n0*8, axis=c, rep=2)
		net = MaxPooling2D(pool_size=(2,2))(net)
		net = unet_conv(net, n0*16, axis=c, rep=2)
		net = UpSampling2D(size=(2,2))(net)
		net = unet_conv(net, n0*8, axis=c, rep=2)
		net = UpSampling2D(size=(2,2))(net)
		net = unet_conv(net, n0*4, axis=c, rep=2)
		net = UpSampling2D(size=(2,2))(net)
		net = unet_conv(net, n0*2, axis=c, rep=2)
		net = UpSampling2D(size=(2,2))(net)
		net = unet_conv(net, n0, axis=c, rep=1)
		labels = Convolution2D(1, (3, 3), activation='sigmoid', padding='same', name='labels')(net)
	elif version == 2:
		n0 = 8
		net = unet_conv(inputs, n0, axis=c, rep=3)
		net = MaxPooling2D(pool_size=(2,2))(net)
		cov2 = unet_conv(net, n0*2, axis=c, rep=2)
		net = MaxPooling2D(pool_size=(2,2))(cov2)
		cov3 = unet_conv(net, n0*4, axis=c, rep=2)
		net = MaxPooling2D(pool_size=(2,2))(cov3)
		cov4 = unet_conv(net, n0*8, axis=c, rep=2)
		net = UpSampling2D(size=(2,2))(cov4)
		net = unet_conv(net, n0*4, axis=c, rep=2)
		net = unet_conv(concatenate([net,cov3], axis=-1), n0*4, axis=c, rep=1)
		net = UpSampling2D(size=(2,2))(net)
		net = unet_conv(net, n0*2, axis=c, rep=2)
		net = unet_conv(concatenate([net,cov2], axis=-1), n0*2, axis=c, rep=1)
		net = UpSampling2D(size=(2,2))(net)
		net = unet_conv(net, n0, axis=c, rep=1)
		labels = Convolution2D(1, (3, 3), activation='sigmoid', padding='same', name='labels')(net)
	elif version == 3:
		sub_rep = cfg_dict['subsampling_conv_repeat'] # number fo time conv2d/BN layers are repeated on the way down
		sup_rep = cfg_dict['upsampling_conv_repeat']  # number of time conv2d/BN layers are repeated on the way up
		kwargs = { 'axis' : c, 'rep' : sub_rep, 'res_like' : cfg_dict.get('resnet_like', False) }
		n0 = 8                                     # (?, 512, 512, 1)
		cov1 = unet_conv(inputs, n0,    **kwargs)  # -> (?, 512, 512, 8)
		net  = MaxPooling2D(pool_size=(2,2))(cov1) # -> (?, 256, 256, 8)
		cov2 = unet_conv(net,    n0*2,  **kwargs)  # -> (?, 256, 256, 16)
		net  = MaxPooling2D(pool_size=(2,2))(cov2) # -> (?, 128, 128, 16)
		cov3 = unet_conv(net,    n0*4,  **kwargs)  # -> (?, 128, 128, 32)
		net  = MaxPooling2D(pool_size=(2,2))(cov3) # -> (?, 64, 64, 32)
		cov4 = unet_conv(net,    n0*8,  **kwargs)  # -> (?, 64, 64, 64)
		net  = MaxPooling2D(pool_size=(2,2))(cov4) # -> (?, 32, 32, 64)
		cov5 = unet_conv(net,    n0*16, **kwargs)  # -> (?, 32, 32, 128)
		# 5
		kwargs['rep'] =sup_rep
		net = UpSampling2D(size=(2,2))(cov5)  # -> (?, 64, 64, 128)
		net = unet_conv(net, n0*8, **kwargs)  # -> (?, 64, 64, 64)
		net = UpSampling2D(size=(2,2))(net)   # -> (?, 128, 128, 64)
		net = unet_conv(net, n0*4, **kwargs)  # -> (?, 128, 128, 32)
		net = UpSampling2D(size=(2,2))(net)   # -> (?, 256, 256, 32)
		net = unet_conv(net, n0*2, **kwargs)  # -> (?, 256, 256, 16)
		net = UpSampling2D(size=(2,2))(net)   # -> (?, 512, 512, 16)
		net5 = unet_conv(net, n0, **kwargs)   # -> (?, 512, 512, 8)
		# 4
		net = UpSampling2D(size=(2,2))(cov4)  # -> (?, 128, 128, 64)
		net = unet_conv(net,  n0*4, **kwargs) # -> (?, 128, 128, 32)
		net = UpSampling2D(size=(2,2))(net)   # -> (?, 256, 256, 32)
		net = unet_conv(net,  n0*2, **kwargs) # -> (?, 256, 256, 16)
		net = UpSampling2D(size=(2,2))(net)   # -> (?, 512, 512, 16)
		net4 = unet_conv(net, n0,   **kwargs) # -> (?, 512, 512, 8)
		# 3
		net = UpSampling2D(size=(2,2))(cov3)  # -> (?, 256, 256, 32)
		net = unet_conv(net,  n0*2, **kwargs) # -> (?, 256, 256, 16)
		net = UpSampling2D(size=(2,2))(net)   # -> (?, 512, 512, 16)
		net3 = unet_conv(net, n0,   **kwargs) # -> (?, 512, 512, 8)

		net = concatenate([net5,net4,net3], axis=-1) # -> (?, 512, 512, 24)
		labels = Convolution2D(1, (1, 1), activation='sigmoid', padding='same', name='labels')(net) # -> (?, 512, 512, 1)
	elif version == 4:
		n0 = 8                                      # (?, 512, 512, 1)
		cov1 = unet_conv(inputs, n0, axis=c, rep=1) # -> (?, 512, 512, 8)
		net  = unet_conv_down(cov1, n0)             # -> (?, 256, 256, 8)
		cov2 = unet_conv(net,  n0*2, axis=c, rep=1) # -> (?, 256, 256, 16)
		net  = unet_conv_down(cov2, n0*2)           # -> (?, 128, 128, 16)
		cov3 = unet_conv(net,  n0*4, axis=c, rep=1) # -> (?, 128, 128, 32)
		net  = unet_conv_down(cov3, n0*4)           # -> (?, 64, 64, 32)
		cov4 = unet_conv(net,  n0*8, axis=c, rep=1) # -> (?, 64, 64, 64)
		net  = unet_conv_down(net, n0*8)            # -> (?, 32, 32, 64)
		cov5 = unet_conv(net, n0*16, axis=c, rep=1) # -> (?, 32, 32, 128)
		# 5
		net = UpSampling2D(size=(2,2))(cov5)        # -> (?, 64, 64, 128)
		net = unet_conv(net, n0*8, axis=c, rep=1)   # -> (?, 64, 64, 64)
		net = UpSampling2D(size=(2,2))(net)         # -> (?, 128, 128, 64)
		net = unet_conv(net, n0*4, axis=c, rep=1)   # -> (?, 128, 128, 32)
		net = UpSampling2D(size=(2,2))(net)         # -> (?, 256, 256, 32)
		net = unet_conv(net, n0*2, axis=c, rep=1)   # -> (?, 256, 256, 16)
		net = UpSampling2D(size=(2,2))(net)         # -> (?, 512, 512, 16)
		net5 = unet_conv(net, n0, axis=c, rep=1)    # -> (?, 512, 512, 8)
		# 4
		net = UpSampling2D(size=(2,2))(cov4)        # -> (?, 128, 128, 64)
		net = unet_conv(net, n0*4, axis=c, rep=1)   # -> (?, 128, 128, 32)
		net = UpSampling2D(size=(2,2))(net)         # -> (?, 256, 256, 32)
		net = unet_conv(net, n0*2, axis=c, rep=1)   # -> (?, 256, 256, 16)
		net = UpSampling2D(size=(2,2))(net)         # -> (?, 512, 512, 16)
		net4 = unet_conv(net, n0, axis=c, rep=1)    # -> (?, 512, 512, 8)
		# 3
		net = UpSampling2D(size=(2,2))(cov3)        # -> (?, 256, 256, 32)
		net = unet_conv(net, n0*2, axis=c, rep=1)   # -> (?, 256, 256, 16)
		net = UpSampling2D(size=(2,2))(net)         # -> (?, 512, 512, 16)
		net3 = unet_conv(net, n0, axis=c, rep=1)    # -> (?, 512, 512, 8)

		net = concatenate([net5,net4,net3], axis=-1) # -> (?, 512, 512, 24)
		labels = Convolution2D(1, (1, 1), activation='sigmoid', padding='same', name='labels')(net) # -> (?, 512, 512, 1)
	else:
		raise Exception("not defined net version")

	model = Model(inputs=inputs, outputs=labels)
	return model

# eof vim: set noet ci pi sts=0 sw=4 ts=4:
