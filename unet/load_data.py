from __future__ import division, print_function

import glob
import os
#import sys
#import time

#from skimage import exposure,io,transform
#import h5py
import numpy as np
float32_t = np.float32

import config as cfg

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

class ImgStream(object):
    def __init__(self, dataset, batch_size, unlabeled_ratio = 1.0):#how many images do we sample from the unlablled images
        self.batch_size = batch_size
        self.split_id = None
        self.unlabeled_ratio = unlabeled_ratio
        if dataset == 'train':
            self.imgs, self.labels, self.split_id = get_file_list('train')
            self.labels = np.asarray(self.labels, dtype=np.uint8)
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

    def CV_fold_gen(self, fold, NCV, shuffleTrain=True):
        """
        returns a generator that iterate the expected set
        returns train_gen, val_gen two generators
        """
        idx = np.arange(self.Tsize)
        assert fold<NCV
        if self.split_id is not None:
            iii = self.split_id%NCV == fold
        else:
            iii = idx%NCV == fold
        train_idx = idx[~iii]
        val_idx = idx[iii]
        return  self._datagen(train_idx, cycle=True, shuffle=shuffleTrain), \
                self._datagen(val_idx, cycle=True, shuffle=False)

    def all_gen(self, cycle=True, shuffle=False, test=False):
        idx = np.arange(self.Tsize)
        return self._datagen(idx, cycle, shuffle, test)

    def _datagen(self, input_idx, cycle, shuffle, istest=False): # infinte data generator
        sel_labels = self.labels[input_idx]
        # x = 1 where sel_labels != 0, self.unlabeled_ratio*np.sum(sel_labels>0)/np.sum(sel_labels==0)
        x = np.ones_like(sel_labels, dtype=float)
        num_labels = np.sum(sel_labels>0)
        num_unlbls = np.sum(sel_labels==0)
        x[sel_labels==0] = self.unlabeled_ratio * num_labels / num_unlbls
        while True:
            # sample from full set of data
            idx = input_idx[np.random.rand(len(x))<x] # all labeled index are included, but only a random fraction of unlabled are picked
            L = len(idx) #dataset size
            Nbatch = L//self.batch_size

            if shuffle:
                np.random.shuffle(idx)
            for b in range(Nbatch):
                batch_idx = idx[b*self.batch_size:min((b+1)*self.batch_size,L)]
                if cycle and len(batch_idx)<self.batch_size:
                    continue
                yield self._load_data(batch_idx, istest)

            if not cycle:
                break

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
        Y_masks = [np.load(self.get_mask(self.imgs[i])) if self.labels[i]>0.5 else np.zeros(X_imgs.shape[1:3],dtype=np.uint8) for i in batch_idx]
        Y_masks = np.array(Y_masks, dtype=float32_t)
        Y_masks = np.reshape(Y_masks,shape)

        # sanity check
        mask_label = [np.any(m) for m in Y_masks]
        assert mask_label == self.labels[batch_idx].tolist(), '{}: {} != {}'.format(batch_idx, mask_label, self.labels[batch_idx])

        return X_imgs, Y_masks

if __name__=='__main__': #test code
    f, l, s = get_file_list(dataset='train')
    from pprint import pformat
    print(pformat([len(f),len(l),len(s)]))
    print(pformat(f[0:10]))
    print(pformat(l[0:10]))
    print(pformat(s[0:10]))
