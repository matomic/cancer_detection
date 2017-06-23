# -*- coding: utf-8 -*-
''' (/^.^(^.^*)> '''
from __future__ import division, print_function

import collections
import glob
import os
#import sys
#import time

#from skimage import exposure,io,transform
import h5py
import numpy as np

from utils import np_rint, float32_t, np_rand, np_shuffle

from pprint import pprint
try:
	from ipdb import set_trace
except Exception:
	pass

ImageMask = collections.namedtuple('ImageMask', ('path', 'has_label', 'subset'))

def get_file_list(imgmask_dir, dataset='train'):
	"""
	for luna16 10 fold data set
	"""
	if dataset=='train':
		out_list = []
		for subset in range(10):
			d = os.path.join(imgmask_dir, 'subset{}'.format(subset))

			files = glob.glob(os.path.join(d, 'image_*.npy'))
			for f in files:
				has_label = os.path.isfile(ImgStream.get_mask(f))
				out_list.append(ImageMask(path=f, has_label=has_label, subset=subset))
		return out_list
	else:
		raise NotImplementedError(dataset)

TrainValidDataset = collections.namedtuple('_CVDS', ('train', 'validation', 'train_size', 'validation_size'))

class ImgStreamBase(object):
	''' (/｡>‿‿<｡(˶′◡‵˶)> '''
	normalization = 1.0 / 255
	def __init__(self, images, labels, findex, batch_size, unlabeled_ratio):
		self.batch_size = batch_size
		self.unlabeled_ratio = unlabeled_ratio

		assert len({len(x) for x in [images, labels, findex]}) == 1, 'images, labels, and findex must be the same length'
		self.images = images # images or paths to image
		self.labels = np.asarray(labels, dtype=np.uint8) # has-label
		self.findex = np.asarray(findex, dtype=np.uint8)  # fold index

		self.total_size = len(self.images)
		self.labeled_size = np.sum(self.labels != 0)
		self.size = int(self.labeled_size* (1 + unlabeled_ratio)) # estimated of stream size

		print('labels: {}'.format(set(self.labels)))
		print("labeled image: {}".format(self.labeled_size))
		print("image data size: {}".format(self.total_size))

	def partitionTrainValidation(self, fold, folds=None):
		'''
		Return two list of indices for training and validation set of {fold} in {folds}
		:type fold:  int
		:type folds: int|None
		'''
		where_valid = (self.findex == fold) if folds is None else (self.findex - fold) % folds == 0
		train_idx = np.where(~where_valid)[0]
		valid_idx = np.where( where_valid)[0]
		return train_idx, valid_idx

	def samplingProbability(self, data_idx):
		'''Given {input_idx} list of input data index, return list of probability each will be sampled:
		 - The probability for index with labeled data is 1; so the expected
		   number of labled indices to be generated is the number of labeled
		   indices.
		 - The probability for index with unlabeled data unlabeled_ratio *
		   |labeled| / |unlabeled|; so the expected number of unlabled indices
		   is unlabeled_ratio * expected number of labeled indices.
		'''
		data_label = self.labels[data_idx] # self.labels should be cast to np.ndarray(dtype=bool) for this to work.
		p_sampling = np.ones_like(data_label, dtype=float)
		unlabeled  = data_label == 0
		num_unlbls = np.sum( unlabeled)
		num_labels = data_label.size - num_unlbls
		p_sampling[unlabeled] = min(1, self.unlabeled_ratio * num_labels / num_unlbls)
		return p_sampling

	def _load_data(self, batch_idx, test=False):
		'''yield batch by {batch_idx}'''
		raise NotImplementedError

	@classmethod
	def normalize_image(cls, image, offset=-0.05):
		'''Normalize a grayscale {image} to ~0 mean ~1 norm'''
		return np.asarray(image, dtype=float32_t)*cls.normalization + offset # to range [0, 1] - 0.05 (mean)

	def _datagen(self, input_idx, cycle, shuffle, test=False):
		'''Infinite data generator'''
		sampling_prop = self.samplingProbability(input_idx)
		while True:
			# Sample from input set
			sample_idx = input_idx[np_rand(input_idx.size) < sampling_prop]
			sample_size = len(sample_idx)
			N_batch = sample_size // self.batch_size
			if shuffle:
				np_shuffle(sample_idx)
			# NOTE: this can be split up per GPU
			for b in range(N_batch):
				batch_slice = slice(b*self.batch_size, (b+1)*self.batch_size)
				batch_idx = sample_idx[batch_slice]
				if cycle and len(batch_idx) < self.batch_size:
					continue # do not yield remainder batch
				yield self._load_data(batch_idx, test=test)
			if not cycle:
				break

	def CV_fold_gen(self, fold, folds, cycle=True, shuffleTrain=True, augmentTrain=None):
		'''Return training and valdation dataset for {fold} of {folds}'''
		train_idx, valid_idx = self.partitionTrainValidation(fold, folds)
		train_gen = self._datagen(train_idx, cycle=cycle, shuffle=shuffleTrain)
		valid_gen = self._datagen(valid_idx, cycle=cycle, shuffle=False)
		train_size = int(np_rint(np.sum(self.samplingProbability(train_idx))))
		valid_size = int(np_rint(np.sum(self.samplingProbability(valid_idx))))
		if augmentTrain:
			train_gen = augmentTrain(train_gen)
		return TrainValidDataset(train=train_gen, validation=valid_gen,
				train_size=train_size, validation_size=valid_size)

	def all_gen(self, cycle=True, shuffle=False, test=False):
		'''Return TrainValidDataset where all data are used for training
		'''
		all_idx = np.arange(self.total_size)
		train_gen = self._datagen(all_idx, cycle, shuffle, test)
		train_size = int(np_rint(np.sum(self.samplingProbability(all_idx))))
		return TrainValidDataset(train_gen, None, train_size, 0)


class ImgStream(ImgStreamBase):
	'''
	>>> train_set = ImgStream('data_root/img_mask_no_lung_mask/', 'train', 10, unlabeled_ratio=0.5)
	labels: set([0, 1])
	labeled image : 3284
	image data size  82089
	>>> tds = train_set.all_gen(cycle=False, shuffle=False, test=False)
	>>> labels = [ np.any(y) for X, Y in tds.train for y in Y]
	>>> ratio = np.count_nonzero(labels) / len(labels)
	>>> ratio # doctest: +ELLIPSIS
	0.66...
	'''

	def __init__(self, imgmask_dir, dataset, batch_size, unlabeled_ratio = 1.0):
		if dataset == 'train':
			image_label_list = get_file_list(imgmask_dir, 'train')

			images = []
			labels = []
			findex = []
			for path, label, subset in image_label_list:
				images.append(path)
				labels.append(label)
				findex.append(subset)
		elif dataset == 'test':
			raise NotImplementedError(dataset)
		else:
			raise NotImplementedError(dataset)

		super(ImgStream, self).__init__(images, labels, findex, batch_size=batch_size, unlabeled_ratio=unlabeled_ratio)

	@staticmethod
	def get_mask(filename):
		'''Return corresponding mask npy file'''
		a, _, c = filename.rpartition('image_')
		return a+'mask_'+c

	def _load_batch_image(self, batch_idx):
		return [np.load(self.images[i]) for i in batch_idx]

	def _load_batch_label(self, batch_idx, batch):
		return [ np.load(self.get_mask(self.images[i]))   # mask
				if bool(self.labels[i]) else              # or
				np.zeros(batch.shape[1:3],dtype=np.uint8) # zeros
				for i in batch_idx]

	def _load_data(self, batch_idx, test=False):
		batch = self._load_batch_image(batch_idx)
		batch = self.normalize_image(batch)
		batch = np.expand_dims(batch, axis=-1)
		if test:
			return batch
		labels = self._load_batch_label(batch_idx, batch)
		labels = np.reshape(np.asarray(labels, dtype=float32_t), batch.shape)

		# sanity check
		mask_label = [np.any(m) for m in labels]
		assert mask_label == self.labels[batch_idx].tolist(), '{}: {} != {}'.format(batch_idx, mask_label, self.labels[batch_idx])
		return batch, labels


class Hdf5ImageStream(ImgStreamBase):
	'''Generates Image stream from HDF5 file'''
	# http://stackoverflow.com/a/27713489/2333932
	# HDF5 file should have the following groups/datasets
	# /images/{images_index} -> 2D array of images (input)
	# /masks/{masks_index}   -> 2D array of mask (output)
	# /labels                -> boolean array, whether the i-th image is labeled.
	def __init__(self, h5file, batch_size, unlabeled_ratio = 1.0):
		self.split_id = None
		self.unlabeled_ratio = unlabeled_ratio
		self.h5file = h5py.File(h5file, 'r')
		self.normaliztion = 1 / 255 # assumes that image array values are uint8

		self.total_size = len(self.h5file['images'])
		size = np.count_nonzero(self.h5file['labels']) * (1 + self.unlabeled_ratio)
		super(Hdf5ImageStream, self).__init__(size=size, batch_size=batch_size)

	def all_gen(self, cycle=True, shuffle=False, test=False):
		'''Generate all the data'''
		return TrainValidDataset(self._datagen(None, cycle, shuffle, test), None, self.size, 0)

	def samplingProbability(self, data_idx):
		datasets = self.getImageBatch(data_idx)
		size = len(datasets)
		sampling_prop  = np.ones(size, dtype=float) # Probability an index is selected
		haslabel   = np.asarray(ds.attrs['haslabel'] for ds in datasets)
		num_labels = np.count_nonzero(haslabel)
		num_unlbls = len(haslabel) - num_labels
		sampling_prop[~haslabel] = self.unlabeled_ratio * num_labels / num_unlbls # generally num_unlbls > num_labels
		return sampling_prop

	def _load_batch_image(self, batch_idx):
		return [self.h5file['/images/{:d}'.format(i)] for i in batch_idx]

	def _load_batch_label(self, batch):
		return [ self.h5file[x.attrs['mask_name']] # mask
				if x.attrs['haslabel'] else        # or
				np.zeros_like(x, dtype=x.dtype)    # zero
				for x in batch]

	def _load_data(self, batch_idx, test=False):
		'''
		#>>> stream = Hdf5ImageStream("/var/data/mlfzhang/LUNA/data_root/img_mask/luna06.h5", 10, 0.5)
		#>>> x, y = stream._load_data([0, 1, 2, 3, 4])
		#>>> x.shape
		#(5, 512, 512, 1)
		#>>> y.shape
		#(5, 512, 512, 1)
		'''
		batch = self._load_batch_image(batch_idx)
		batch = self.normalize_image(batch)
		batch = np.expand_dims(batch, axis=-1)
		if test:
			return batch
		labels = self._load_batch_label(batch)
		labels = np.asarray(labels, shape=batch.shape, dtype=float32_t)
		# sanity check
		mask_label = [np.any(m) for m in labels]
		assert mask_label == [x.attrs['haslabel'] for x in batch]
		return batch, labels

	def getImageBatch(self, batch_idx=None, assert_valid_index=True):
		'''Return image dataset list by index
		:rtype: list
		'''
		if batch_idx is None:
			return self.h5file['/images'].values()
		batch = []
		for i in batch_idx:
			index = '{:d}'.format(i)
			if index in self.h5file['/images']:
				batch.append(self.h5file['/images'][index])
			elif assert_valid_index:
				raise IndexError(i)
			else:
				print("image index {:d} not found")
		return batch

if __name__=='__main__': #test code
	#f, l, s = get_file_list(dataset='train')
	#from pprint import pformat
	#print(pformat([len(f),len(l),len(s)]))
	#print(pformat(f[0:10]))
	#print(pformat(l[0:10]))
	#print(pformat(s[0:10]))
	import doctest
	doctest.testmod()

# eof vim: set noet ci pi sts=0 sw=4 ts=4:
