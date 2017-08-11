'''train'''
from __future__ import print_function
from __future__ import division

# stdlib
import functools
import os
import sys
import argparse

# 3rd-party
import numpy as np
import simplejson as json
#import pandas as pd
#import tensorflow as tf

from keras import callbacks, optimizers as opt
from keras.backend import tensorflow_backend as K
from keras.models import load_model
#from keras.metrics import categorical_accuracy
#from keras.objectives import categorical_crossentropy

# in-house
from config import UnetConfig
from console import PipelineApp
from img_augmentation import ImageAugment
from load_data import ImgStream
from losses import get_loss
from models import get_net, transfer_weight
from utils import hist_summary

# DEBUGGING
from pprint import pprint
try:
	from ipdb import set_trace
except Exception:
	pass


def fit_callbacks(chkpt_path):
	'''Return call back functions during fitting'''
	## call backs
	model_checkpoint = callbacks.ModelCheckpoint(
	    chkpt_path,
	    monitor='val_loss', verbose=0,
	    save_best_only=True
	    )
	learn_rate_decay = callbacks.ReduceLROnPlateau(
	    monitor='val_loss',factor=0.3,
	    patience=5, min_lr=1e-4, verbose=1
	    )
	earlystop = callbacks.EarlyStopping(monitor='val_loss',patience=10,mode='min',verbose=1)
	#tensorboard = callbacks.TensorBoard(log_dir=cfg.log_dir,histogram_freq=10,write_graph=False)
	# TODO: add model synchronization at epoch end for training multiple models, e.g. across device
	#return [learn_rate_decay, model_checkpoint, earlystop, tensorboard]

	return [
	    learn_rate_decay,
	    model_checkpoint,
	    earlystop
	    ]

def load_unet(cfg, checkpoint_path=None):
	'''Load and compile model from scratch or from {checkpoint_path}'''
	loss = get_loss(cfg.loss_func, cfg.loss_args)
	if checkpoint_path:
		if isinstance(loss, str):
			custom_objects = {}
		else:
			custom_objects = {
			    loss.__name__ : loss
			}
		model = load_model(checkpoint_path, custom_objects=custom_objects, compile=False)
		print("model loaded from {}".format(checkpoint_path))
	else:
		model = get_net(cfg.net)
		print("model loaded from scratch")
	model.compile(
	    optimizer = getattr(opt, cfg.fitter['opt'])(**cfg.fitter['opt_arg']),
	    loss = loss
	    )
	return model

def load_model_from_session(session_json_path, fold='0'):
	'''Load unet model from session at {session_json_path}'''
	session_json = json.load(open(session_json_path, 'r'))
	session_cfg  = UnetConfig.fromDict(session_json['unet']['config'])
	checkpt_path = os.path.join(os.path.dirname(session_json_path),
			session_json['unet']['models'][str(fold)]['checkpoint_path'])
	return load_unet(session_cfg, checkpt_path)


class UnetTrainer(PipelineApp):
	'''Train Unet model with simple 2D slices'''
	def __init__(self):
		super(UnetTrainer, self).__init__()
		self.full_train_data = []
		self.fitter = None

	def arg_parser(self):
		parser = super(UnetTrainer, self).arg_parser()
		parser = argparse.ArgumentParser(add_help=True,
		    description='Train Unet',
		    parents=[parser], conflict_handler='resolve')

		parser.add_argument('--init-weight-from', action='store',
				help='Initialize weight by transfering weights from another session')

		parser.add_argument('--debug', action='store_true',
		    help='enter debug model')

		return parser

	def argparse_postparse(self, parsedArgs=None):
		super(UnetTrainer, self).argparse_postparse(parsedArgs)
		init_weight_from = parsedArgs.init_weight_from
		if init_weight_from:
			if not os.path.isdir(init_weight_from):
				init_weight_from = os.path.join(self.dirs.res_dir, init_weight_from)
			if os.path.isdir(init_weight_from):
				parsedArgs.init_weight_from = init_weight_from
			else:
				parsedArgs.init_weight_from = None
				print("WARN: cannot determine session of --init-weights-from {}.".format(parsedArgs.init_weight_from))
		self.fitter = self.unet_cfg.fitter
		# Normally, img_mask_gen.py generates image/nodule mask pairs in self.cfg.root/img_mask, so that training data are reusable.  However,
		# sometimes we might want to experiment with different preprocessing approaches.  So here we expect training data to come from the same place
		# as where we will be saving the result.  To use "standard" training data, make a symbolic link from result_dir back to root dir.
		img_mask_dir = os.path.join(self.result_dir, 'img_mask')
		assert os.path.isdir(img_mask_dir), 'image/nodule mask pairs are expected in {}. It does not exist'.format(img_mask_dir)
		assert self.unet_cfg.net.CHANNEL in {1,3}, 'Unsupported CHANNEL: {}'.format(self.unet_cfg.CHANNEL)
		assert self.unet_cfg.net.DEPTH in {None, 1, 3}, 'Unsupported DEPTH: {}'.format(self.unet_cfg.DEPTH)
		if self.unet_cfg.net.CHANNEL != 1:
			use_neighbor = 'as_channel'
		elif self.unet_cfg.net.DEPTH != 1:
			use_neighbor = 'as_depth'
		else:
			use_neighbor = False
		self.full_train_data = ImgStream(img_mask_dir, "train",
		    batch_size      = self.fitter['batch_size'],
		    use_neighbor    = use_neighbor,
		    unlabeled_ratio = self.unet_cfg.unlabeled_ratio)

	def _main_impl(self):
		if self.parsedArgs.debug:
			self.do_debug()
		else:
			self.do_train()

	def get_train_data(self, fold, NCV, augmentTrain=None):
		'''Return training data generators for {fold}
		{augmentTrain}, if not None, should be a function that takes a generator of training data and returns another generator of augmented input data.
		'''
		if fold < NCV:# CV
			dataset = self.full_train_data.CV_fold_gen(fold, NCV, shuffleTrain=True, augmentTrain=augmentTrain)
		else:#all training data
			dataset = self.full_train_data.all_gen(shuffle=True)
		return dataset

#	def trainfor(model, tds, epochs, steps_per_epoch, chkpt_path, fitter):
#		'''train {model} with training dataset {tds} for {epochs}'''
#		batch_size = fitter['batch_size']
#		steps_per_epoch = ((steps_per_epoch or 3000) // batch_size) * batch_size
#		history = model.fit_generator(tds.train,
#		    steps_per_epoch  = steps_per_epoch,
#		    epochs           = epochs,
#		    validation_data  = tds.validation,
#		    validation_steps = tds.size // fitter['NCV'],
#		    callbacks        = fit_callbacks(chkpt_path)
#		    )
#		return history

	checkpoint_path_fmrstr = '{net}_{WIDTH}_{tag}_fold{fold}.hdf5'

	def do_train(self):
		"""
		build and train the CNNs.
		"""
		np.random.seed(1234) # pylint: disable=no-member

		sess = K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
		#image augmentation, flow
		aug_kwds = self.unet_cfg.aug
		imgAug = functools.partial(ImageAugment(**aug_kwds).flow_gen, mode='fullXY')

		NCV = self.fitter['NCV']
		batch_size = self.fitter['batch_size']

		folds_best_epoch = []
		folds_history = {}
		for fold in self.fitter['folds']:
			assert fold <= NCV
			print("--- CV for fold: {}".format(fold))

			tds = self.get_train_data(fold, NCV, augmentTrain=imgAug)

			if fold < NCV:#CV
				num_epochs = self.fitter['num_epochs']
			else:
				assert False, 'bad branch'
				num_epochs = int(np.mean(folds_best_epoch)+1) if folds_best_epoch else self.fitter['num_epochs']

			train_dict = self.session_json['unet'].setdefault('models', {}).setdefault(fold, {})
			if 'checkpoint_path' in train_dict:
				checkpoint_path = os.path.join(self.result_dir, train_dict['checkpoint_path'])
			else:
				checkpoint_path = self.checkpoint_path('unet', fold, WIDTH=self.unet_cfg.net.WIDTH, tag=self.unet_cfg.tag)

			with K.tf.device('/gpu:0'):
				K.set_session(sess)
				if os.path.isfile(checkpoint_path):
					model = load_unet(self.unet_cfg, checkpoint_path)
				else:
					model = load_unet(self.unet_cfg)
				model.summary()
				model.get_config()

			## Transfer weights from flat unet
			if self.parsedArgs.init_weight_from:
				old_model = load_model_from_session(os.path.join(self.parsedArgs.init_weight_from, 'session.json'), fold=fold)
				transfer_weight(old_model, model)

			# Save configuration and model to JSON before training
			train_dict['checkpoint_path'] = os.path.basename(checkpoint_path)
			self.dump_session()

			with K.tf.device('/gpu:0'):
				K.set_session(sess)
				num_epochs = 40

				set_trace()

				history = model.fit_generator(tds.train,
				    steps_per_epoch  = (tds.train_size // batch_size),
				    epochs           = num_epochs,
				    validation_data  = tds.validation,
				    validation_steps = tds.validation_size // NCV,
				    callbacks        = fit_callbacks(checkpoint_path)
				    )

			#history has epoch and history atributes
			train_dict['history'] = folds_history[fold] = history.history
			folds_best_epoch.append(np.argmin(history.history['val_loss']))
			self.dump_session()
			del model

		##save the full fitting history file for later study
		#with open(os.path.join(self.dirs.log_dir,'history_{}_{}.pkl'.format(self.cfg.WIDTH, self.cfg.tag)),'wb') as f:
		#    pickle.dump(folds_history, f)

		#summary of all folds
		summary = hist_summary(folds_history)
		print(summary)

	def do_debug(self):
		'''run this when --debug is specified'''
		sess = K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
		K.set_session(sess)
		#chkpt_path = self.parsedArgs.chkpt_path or 'debug_checkpoint.h5'
		fitter = self.unet_cfg.fitter
		batch_size = fitter['batch_size']
		fold = 0

		train_dict = self.session_json['unet'].setdefault('models', {}).setdefault(fold, {})
		if 'checkpoint_path' in train_dict:
			checkpoint_path = os.path.join(self.result_dir, train_dict['checkpoint_path'])
		else:
			checkpoint_path = self.checkpoint_path('unet', fold, WIDTH=self.unet_cfg.net.WIDTH, tag=self.unet_cfg.tag)

		if os.path.isfile(checkpoint_path):
			model = load_unet(self.unet_cfg, checkpoint_path)
		else:
			model = load_unet(self.unet_cfg)
		model.summary()

		set_trace()

		#a_func = functools.partial(ImageAugment(**config.aug).flow_gen, mode='fullXY')
		tds = self.full_train_data.CV_fold_gen(fold=0, folds=10, cycle=False, shuffleTrain=False)
		#tds = self.full_train_data.all_gen(cycle=False, shuffle=False)

		from keras.losses import binary_crossentropy

		x_imag_tf = K.tf.placeholder(shape=(None, 512, 512, 1), dtype=K.tf.float32)
		y_true_tf = K.tf.placeholder(shape=(None, 512, 512, 1), dtype=K.tf.float32)
		y_pred_tf = model(x_imag_tf)
		dice_kwargs = {'per_sample': True, 'smooth': 5e-1 }
		dc_kwargs = { 'pred_weight' : 0.5, **dice_kwargs}
		loss_1 = get_loss('weighted_dice_coef_loss', dc_kwargs)(y_true_tf, y_pred_tf)
		loss_2 = get_loss('dice_coef_loss', {'pred_mul' : 1.0, **dice_kwargs})(y_true_tf, y_pred_tf)
		#loss_3 = get_loss('dice_coef_loss')(y_true_tf, y_pred_tf, smooth=1e-5, p_ave=0)
		#loss_4 = get_loss('dice_coef_loss')(y_true_tf, y_pred_tf, smooth=1e-5, p_ave=1)
		xe_kwargs = { 'pos_weight' : 1.0, 'axis' : (1,2,3) }
		loss_3 = get_loss('weighted_binary_crossentropy')(y_true_tf, y_pred_tf, **xe_kwargs)
		loss_4 = K.mean(binary_crossentropy(y_true_tf, y_pred_tf), axis=(1,2))
		loss_5 = get_loss('combination_loss')(y_true_tf, y_pred_tf, xe_kwargs=xe_kwargs, dc_kwargs=dc_kwargs)

		out_dict = {
				'tp' : [],
				'tn' : [],
				'fp' : [],
				'fn' : [],
				}

		count = 0
		for n, (x, y) in enumerate(tds.validation):
			l1_b, l2_b, l3_b, l4_b, l5_b, y_pred = sess.run([loss_1, loss_2, loss_3, loss_4, loss_5, y_pred_tf],
			    feed_dict = {x_imag_tf: x, y_true_tf: y, K.learning_phase(): False})
			for y_t, y_p, l1, l2, l3, l4, l5 in zip(y, y_pred, l1_b, l2_b, l3_b, l4_b, l5_b):
				is_true = np.any(y_t > 0.5)
				pd_true = np.any(y_p > 0.5)
				out = [l1, l2, l3, l4, l5]
				if pd_true:
					if np.any(y_p * y_t > 0.5):
						out_dict['tp'].append(out)
					else:
						out_dict['fp'].append(out)
				elif is_true:
					out_dict['fn'].append(out)
				else:
					out_dict['tn'].append(out)

				count += 1
				print(("{:3d}, batch {:3d}: "
				    "dice_coeff_loss': {:g}; "
				    "dice_coeff_loss: {:g}; "
				    "crossentropy: {:g}({:g}).")
				    .format(count, n, l1, l2, l3, l4))
#			if n == 10:
#				break
		out_dict = { k : np.asarray(v) for k, v in out_dict.items() }

		plot_kwargs = {
				'tp' : { 'marker' : '*', 'label' : 'T+ ({:2d})' },
				'tn' : { 'marker' : 'o', 'label' : 'T- ({:2d})' },
				'fp' : { 'marker' : '+', 'label' : 'F+ ({:2d})' },
				'fn' : { 'marker' : 'x', 'label' : 'F- ({:2d})' },
				}

		try:
			from matplotlib import pyplot as plt
			plt.ion()
			plt.figure(1, figsize=(9,6))
			plt.clf()
			ix, iy = 0, 2
			for g in ['tp', 'tn', 'fp', 'fn']:
				x = out_dict[g][:,ix]
				y = out_dict[g][:,iy]
				m = plot_kwargs[g]['marker']
				plt.plot(x, y, m, label=plot_kwargs[g]['label'].format(x.size))
			plt.legend()
			plt.xlabel('dice-coeff')
			plt.ylabel('xentropy')
			plt.title(r'$\epsilon={}, \lambda={}, k={}$'.format(dc_kwargs['smooth'], dc_kwargs['pred_weight'], xe_kwargs['pos_weight']))

			plt.figure(2, figsize=(9,6))
			plt.clf()
			for g in ['tp', 'tn', 'fp', 'fn']:
				x = out_dict[g][:,ix]
				y = x + (5 * out_dict[g][:,iy])**(1/2)
				m = plot_kwargs[g]['marker']
				plt.plot(x, y, m, label=plot_kwargs[g]['label'].format(x.size))
			plt.legend()
			plt.xlabel('dice-coeff')
			plt.ylabel('combination')
			plt.title(r'$\epsilon={}, \lambda={}, k={}$'.format(dc_kwargs['smooth'], dc_kwargs['pred_weight'], xe_kwargs['pos_weight']))
		except:
			set_trace()
			raise
		else:
			set_trace()

if __name__=='__main__':
	sys.exit(UnetTrainer().main() or 0)

# eof vim: set noet ci pi sts=0 sw=4 ts=4:
