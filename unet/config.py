'''Defines source-agnostic configuration structure'''
import collections
import simplejson as json

from utils import safejsondump

def namedtuple_with_defaults(typename, field_names, default_values=()):
	T = collections.namedtuple(typename, field_names)
	T.__new__.__defaults__ = (None,)*len(T._fields)
	if default_values:
		if isinstance(default_values, collections.Mapping):
			proto = T(**default_values)
		else:
			proto = T(*default_values)
		T.__new__.__defaults__ = tuple(proto)
	return T

class SerializableMixin(namedtuple_with_defaults('SerializableNamedtuples', ())):
	'''namedtuple Mixin with ability to serialize'''
	@classmethod
	def fromDict(cls, dic):
		return cls(**{ k : dic.get(k) for k in cls._fields})

	def toDict(self):
		return { k : getattr(self, k) for k in self._fields }

	@classmethod
	def fromJson(cls, json_path):
		json_data = json.load(open(json_path, 'r'))
		return cls.fromDict(json_data)

	def toJson(self, json_path):
		safejsondump(self.toDict(), json_path)

	@classmethod
	def fromConfigPy(cls, cfg):
		if not cfg.startswith('config_'):
			cfg = 'config_' + cfg
		return cls.fromDict(__import__(cfg).__dict__)


class NetSpecs(namedtuple_with_defaults('NetSpec',
	('name', 'version', 'WIDTH', 'HEIGHT', 'DEPTH', 'CHANNEL', 'params')), SerializableMixin):
	'''Specifies neural networks'''
NetSpecs.__new__.__default__ = (None,)*len(NetSpecs._fields)



class UnetConfig(collections.namedtuple('UnetConfig', (
	'tag',
	'net',
	'keep_prob',
	'unlabeled_ratio',
	'loss_func',
	'loss_args',
	'fitter',
	#'WIDTH',
	#'HEIGHT',
	#'CHANNEL',
	'aug',
	#'use_augment'
	)), SerializableMixin):
	'''Configuration data for UNet'''
	@classmethod
	def default(cls):
		'''current default'''
		return cls(
		    tag = 5,
		    net = NetSpecs(
		        name    = 'unet',
		        version = 3,
		        WIDTH   = 512,
		        HEIGHT  = 512,
		        CHANNEL = 1,
		        params  = {
		           'subsampling_conv_repeat' : 2,
		           'upsampling_conv_repeat'  : 1,
		        }
		    ),
		    keep_prob = 0.3,
		    unlabeled_ratio = 0.5,
		    fitter = {
		        'batch_size' : 8,
		        'num_epochs' : 40,
		        'NCV'        : 5,   # nonlinear cross validation
		        'folds'      : [0], #fold==NCV means to use all data
		        'opt'        : 'Adam',
		        'opt_arg'    : { 'lr' : 1e-3 }
		        },
		    #loss function paramters
		    #loss_func = 'dice_coef_loss_gen'
		    #loss_args = {
		    #        'smooth'   : 5,
		    #        'pred_mul' : 0.6,
		    #        'p_ave'    : 0.8
		    #}
		    loss_func = 'binary_crossentropy',
		    loss_args = {},
		    aug = {
		        'featurewise_center'            : False,
		        'samplewise_center'             : False,
		        'featurewise_std_normalization' : False,
		        'samplewise_std_normalization'  : False,
		        'zca_whitening'                 : False,
		        'rotation_range'                : 10.0,
		        'width_shift_range'             : 0.20,
		        'height_shift_range'            : 0.20,
		        'shear_range'                   : 0.05,
		        'zoom_range'                    : 0.1,
		        'channel_shift_range'           : 0.,
		        'fill_mode'                     : 'constant',
		        'cval'                          : 0.,
		        'horizontal_flip'               : True,
		        'vertical_flip'                 : False,
		        'rescale'                       : None,
		        }
		    )

	@classmethod
	def fromDict(cls, dic):
		if 'net' in dic:
			if isinstance(dic['net'], dict):
				dic['net'] = NetSpecs(**dic['net'])
			elif not isinstance(dic['net'], NetSpecs):
				assert False
		return super(UnetConfig, cls).fromDict(dic)


class N3DConfig(collections.namedtuple('N3DConfig', (
	'tag',
	'net',
	'fitter',
	#'WIDTH',
	#'HEIGHT',
	#'CHANNEL',
	'aug',
	)), SerializableMixin):
	'''Configuration data for 3D binary classfication net'''

	@classmethod
	def fromDict(cls, dic):
		if 'net' in dic:
			if isinstance(dic['net'], dict):
				dic['net'] = NetSpecs(**dic['net'])
			elif not isinstance(dic['net'], NetSpecs):
				dic['net'] = NetSpecs(*dic['net'])
		return super(N3DConfig, cls).fromDict(dic)

# eof vim: set noet ci pi sts=0 sw=4 ts=4:
