from __future__ import division, print_function

import collections
import glob
import os
#import sys
#import time

#from skimage import exposure,io,transform
import h5py
import numpy as np

float32_t = np.float32

import config as cfg

from pprint import pprint
#from ipdb import set_trace

def get_file_list(dataset='train'):
    """
    for luna16 10 fold data set
    """
    if dataset=='train':
        fs = []  # file list
        ls = []  # has label list (True/False)
        ids = [] # subset index list
        for s in range(10):
            d = os.path.join(cfg.root,'img_mask','subset{}'.format(s))
            files = glob.glob(os.path.join(d,'image_*.npy'))
            labs = glob.glob(os.path.join(d,'mask_*.npy'))
            fs.extend(files)
            fids = [x.split('/')[-1][6:-4] for x in files]
            lids = {x.split('/')[-1][5:-4] for x in labs}
            ll = [x in lids for x in fids]
            ls.extend(ll)
            ids.extend([s]*len(ll))
        # fs=file list, ls:=1 means there is a mask file, or else no (all zero)
        return fs, ls, ids

TrainValidDataset = collections.namedtuple('_CVDS', ('train', 'validation', 'size'))

class ImgStreamBase(object):
    def __init__(self):
        self.size = None
        self.batch_size = None

    def partitionTrainValidation(self, fold, folds):
        '''
        Return two list of indices for training and validation set of {fold} in {folds}
        '''
        raise NotImplementedError

    def samplingProbability(self, input_idx=None):
        '''Given list of input index generate, return list of probability each will be sampled'''
        raise NotImplementedError

    def _load_data(self, batch_idx, test):
        raise NotImplementedError

    def _datagen(self, input_idx, cycle, shuffle, test=False):
        '''Infinite data generator
        '''
        input_idx, sampling_prop = self.samplingProbability(input_idx)
        while True:
            # Sample from input set
            sample_idx = input_idx[np.random.rand(input_idx.size) < sampling_prop]
            sample_size = len(sample_idx)
            N_batch = sample_size // self.batch_size
            if shuffle:
                np.random.shuffle(sample_idx)
            # NOTE: this can be split up per GPU
            for b in xrange(N_batch):
                batch_slice = slice(b*self.batch_size, (b+1)*self.batch_size)
                batch_idx = sample_idx[batch_slice]
                if cycle and len(batch_idx) < self.batch_size:
                    continue # do not yield remainder batch
                yield self._load_data(batch_idx, test=test)
            if not cycle:
                break

    def CV_fold_gen(self, fold, folds, shuffleTrain=True, augmentTrain=None):
        '''Return training and valdation dataset for {fold} of {folds}'''
        train_idx, valid_idx = self.partitionTrainValidation(fold, folds)
        train_gen = self._datagen(train_idx, cycle=True, shuffle=shuffleTrain)
        valid_gen = self._datagen(valid_idx, cycle=True, shuffle=False)
        if augmentTrain:
            train_gen = augmentTrain(train_gen)
        return TrainValidDataset(train_gen, valid_gen, self.size)

    def all_gen(self, cycle=True, shuffle=False, test=False):
        '''Return TrainValidDataset where all data are used for training
        '''
        return TrainValidDataset(self._datagen(None, cycle, shuffle, test),
                None,
                self.size)


class ImgStream(ImgStreamBase):
    def __init__(self, dataset, batch_size, unlabeled_ratio = 1.0):
        self.batch_size = batch_size
        self.split_id = None
        self.unlabeled_ratio = unlabeled_ratio
        if dataset == 'train':
            self.imgs, self.labels, self.split_id = get_file_list('train')
            self.labels = np.asarray(self.labels, dtype=np.uint8)
            print('labels: {}'.format(set(self.labels)))
            self.split_id = np.asarray(self.split_id,dtype=np.int)
            # select part of the no label images
            idx = self.labels>0
            print("labeled image :", np.sum(idx))
            #idx = np.where(np.logical_or(idx,np.random.rand(len(self.labels))<1.0*np.sum(self.labels)/len(self.labels)))[0]
            #self.imgs = [self.imgs[i] for i in idx]
            #self.labels = [self.labels[i] for i in idx]

            print("image data size ", len(self.imgs))
        elif dataset == 'test':
            raise NotImplementedError(dataset)
        else:
            raise NotImplementedError(dataset)

        self.Tsize = len(self.imgs) # number of images
        self.size = int(np.sum(idx) * (1 + self.unlabeled_ratio)) # number of genreated images

    def __len__(self):
        return self.size

    def partitionTrainValidation(self, fold, folds):
        idx = np.arange(self.Tsize)
        iii = (idx if self.split_id is None else self.split_id) % folds == fold
        train_idx = idx[~iii]
        valid_idx = idx[iii]
        return train_idx, valid_idx

    def samplingProbability(self, input_idx=None):
        if input_idx is None:
            input_idx = np.arange(self.Tsize)
        sel_labels = self.labels[input_idx]
        # x = 1 where sel_labels != 0, self.unlabeled_ratio*np.sum(sel_labels>0)/np.sum(sel_labels==0)
        p_sampling = np.ones_like(sel_labels, dtype=float)
        num_labels = np.sum(sel_labels>0)
        num_unlbls = np.sum(sel_labels==0)
        p_sampling[sel_labels==0] = self.unlabeled_ratio * num_labels / num_unlbls
        return input_idx, p_sampling

    @staticmethod
    def get_mask(filename):
        '''Return corresponding mask npy file'''
        a, _, c = filename.rpartition('image')
        return a+'mask'+c

    def _load_data(self, batch_idx, test=False):
        X_imgs = [np.load(self.imgs[i]) for i in batch_idx]
        X_imgs = np.array(X_imgs, dtype=float32_t)*(1.0/255)-0.05 #to range (0,1) - 0.05 (mean)
        shape  = tuple(list(X_imgs.shape)+[1])
        X_imgs = np.reshape(X_imgs,shape)
        if test:
            return X_imgs
        Y_masks = [ np.load(self.get_mask(self.imgs[i]))   # mask
                if bool(self.labels[i]) else               # or
                np.zeros(X_imgs.shape[1:3],dtype=np.uint8) # zeros
                for i in batch_idx]
        Y_masks = np.array(Y_masks, dtype=float32_t)
        Y_masks = np.reshape(Y_masks,shape)

        # sanity check
        mask_label = [np.any(m) for m in Y_masks]
        assert mask_label == self.labels[batch_idx].tolist(), '{}: {} != {}'.format(batch_idx, mask_label, self.labels[batch_idx])

        return X_imgs, Y_masks

    def all_gen(self, cycle=True, shuffle=False, test=False):
        '''
        >>> train_set = ImgStream("train", 10, unlabeled_ratio=0.5)
        labels: set([0, 1])
        labeled image : 3284
        image data size  82089
        >>> tds = train_set.all_gen(cycle=False, shuffle=False, test=False)
        >>> labels = [ np.any(y) for X, Y in tds.train for y in Y]
        >>> ratio = np.count_nonzero(labels) / len(labels)
        >>> ratio # doctest: +ELLIPSIS
        0.66...
        '''
        return super(ImgStream, self).all_gen(cycle, shuffle, test)


class Hdf5ImageStream(ImgStreamBase):
    '''Generates Image stream from HDF5 file'''
    # http://stackoverflow.com/a/27713489/2333932
    # HDF5 file should have the following groups/datasets
    # /images/{images_index} -> 2D array of images (input)
    # /masks/{masks_index}   -> 2D array of mask (output)
    # /labels                -> boolean array, whether the i-th image is labeled.
    def __init__(self, h5file, batch_size, unlabeled_ratio = 1.0):
        self.batch_size = batch_size
        self.split_id = None
        self.unlabeled_ratio = unlabeled_ratio
        self.h5file = h5py.File(h5file, 'r')
        self.normaliztion = 1 / 255 # assumes that image array values are uint8

        self.total_size = len(self.h5file['images'])
        self.size = np.count_nonzero(self.h5file['labels']) * (1 + self.unlabeled_ratio)

    def __len__(self):
        '''size of the generator'''
        return self.size

    def partitionTrainValidation(self, fold, folds):
        '''
        Create two generators, training and validation, for fold-cross validation
        :type fold:  int
        :type folds: int
        :rtype: TrainValidDataset
        '''
        idx = np.arange(self.total_size)
        iii = (idx - fold) % folds == 0
        train_idx = idx[~iii]
        valid_idx = idx[iii]
        return train_idx, valid_idx

    def all_gen(self, cycle=True, shuffle=False, test=False):
        '''Generate all the data'''
        return TrainValidDataset(self._datagen(None, cycle, shuffle, test), None, self.size)

    def samplingProbability(self, input_idx=None):
        datasets = self.getImageBatch(input_idx)
        size = len(datasets)
        if input_idx is None:
            input_idx = np.arange(size)
        sampling_prop  = np.ones(size, dtype=float) # Probability an index is selected
        haslabel   = np.asarray(ds.attrs['haslabel'] for ds in datasets)
        num_labels = np.count_nonzero(haslabel)
        num_unlbls = len(haslabel) - num_labels
        sampling_prop[~haslabel] = self.unlabeled_ratio * num_labels / num_unlbls # generally num_unlbls > num_labels
        return input_idx, sampling_prop

    def _load_data(self, batch_idx, test=False):
        '''
        >>> stream = Hdf5ImageStream("/var/data/mlfzhang/LUNA/data_root/img_mask/luna06.h5", 10, 0.5)
        >>> x, y = stream._load_data([0, 1, 2, 3, 4])
        >>> x.shape
        (5, 512, 512, 1)
        >>> y.shape
        (5, 512, 512, 1)
        '''
        batch = [self.h5file['/images/{:d}'.format(i)] for i in batch_idx]
        X_imgs = np.array(batch, dtype=float32_t)* self.normaliztion - 0.05
        X_imgs = np.expand_dims(X_imgs, axis=-1)
        if test:
            return X_imgs
        Y_masks = [ self.h5file[x.attrs['mask_name']]
                if x.attrs['haslabel'] else
                np.zeros_like(x, dtype=x.dtype)
                for x in batch]
        Y_masks = np.array(Y_masks, dtype=float32_t)
        Y_masks = np.reshape(Y_masks, X_imgs.shape)
        # sanity check
        mask_label = [np.any(m) for m in Y_masks]
        assert mask_label == [x.attrs['haslabel'] for x in batch]
        return X_imgs, Y_masks

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
