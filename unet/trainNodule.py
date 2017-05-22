# -*- coding: utf-8 -*-
'''[^._.^]ﾉ彡'''
from __future__ import print_function
from __future__ import division

import os
import pickle
import sys

import numpy as np
# from keras import backend as K
from keras import callbacks
from keras import optimizers as opt
from keras.models import load_model

from img_augmentation2 import ImageDataGenerator
from models import get_3Dnet

from console import PipelineApp

# DEBUG
from pprint import pprint
from ipdb import set_trace


npf32_t = np.float32 # pylint: disable=no-member

def model_factory(cfg, chkpt_path):
    '''Load model from {chkpt_path} or build fresh'''
    if chkpt_path is not None and os.path.isfile(chkpt_path):
        model = load_model(chkpt_path)
        print("model loaded from checkpoint {}.".format(chkpt_path))
    else:

        ## build the neural net work
        model = get_3Dnet(cfg.net.name, cfg.net.version, cfg.WIDTH, cfg.HEIGHT, cfg.CHANNEL)
        model.compile(
                optimizer = getattr(opt, cfg.fitter['opt'])(**cfg.fitter['opt_arg']),
                loss = 'binary_crossentropy'
                )
        print("model loaded from scratch")
    set_trace()
    return model

def train(Xtrain, Ytrain, Xval, Yval, cfg, checkpoint_path):
    '''Train model with ({Xtrain}, {Ytrain}), with validation loss from ({Xval}, {Yval})'''
    #call backs
    datagen = ImageDataGenerator(**cfg.aug)
    datagenOne = ImageDataGenerator()

    # load model
    model = model_factory(cfg, checkpoint_path)

    # setup callbacks
    model_checkpoint = callbacks.ModelCheckpoint(checkpoint_path,
            monitor='val_loss', verbose=0,
            save_best_only=False
            )
    learn_rate_decay = callbacks.ReduceLROnPlateau(
            monitor='val_loss',factor=0.3,
            patience=8, min_lr=1e-5, verbose=1
            )
    earlystop = callbacks.EarlyStopping(monitor='val_loss',patience=15,mode='min',verbose=1)

    #Fit here
    batch_size = cfg.fitter['batch_size']
    train_flow = datagen.flow(Xtrain,  Ytrain, batch_size=batch_size)
    valid_flow = datagenOne.flow(Xval, Yval,   batch_size=batch_size)
    history = model.fit_generator(train_flow,
            steps_per_epoch  = len(Xtrain) // batch_size,
            epochs           = cfg.fitter['num_epochs'],
            validation_data  = valid_flow,
            validation_steps = len(Xval) // batch_size,
            callbacks        = [learn_rate_decay, model_checkpoint, earlystop]
            )
    del model
    return history

class TrainNoduleApp(PipelineApp):
    def argparse_postparse(self, parsedArgs=None):
        super(TrainNoduleApp, self).argparse_postparse(parsedArgs)
        assert self.parsedArgs.session, 'unspecified unet training --session'
        self.input_dir = os.path.join(self.result_dir, 'nodule_candidates')
        # output_dir is self.checkpoint_path for this stage.

    def getInputData(self):
        '''Return input data list'''
        np.random.seed(123) # pylint: disable=no-member
        assert os.path.isdir(self.input_dir), 'not a valid dir: {}'.format(self.input_dir)
        print("loading {} ...".format(self.input_dir))
        ids   = os.listdir(self.input_dir)
        data  = [pickle.load(open(os.path.join(self.input_dir, i),'rb')) for i in ids]
        return data

    checkpoint_path_fmrstr = '{net}_{WIDTH}_{tag}_fold{fold}.hdf5'
    def _main_impl(self):
        data  = self.getInputData()
        Ncase = np.sum(len(imgs) for imgs, _ls in data)

        Y  = np.zeros(Ncase)
        W  = self.cfg.WIDTH
        NC = self.cfg.CHANNEL
        X  = np.zeros((len(Y), W, W, NC), dtype=npf32_t) #set to 0 for empty chanels
        c  = 0
        for imgs, lbls in data:
            nx = len(imgs)
            if nx == 0:
                continue
            Y[c:c+nx] = lbls
            X[c:c+nx] = imgs
            c += nx
        print("total training cases ", Ncase)
        print("percent Nodules: ",np.sum(Y)/Ncase)
        set_trace()

        # N fold cross validation
        NF = self.cfg.fitter['NCV']
        ii =  np.arange(Ncase)
        for i in self.cfg.fitter['folds']:
            ival   = ii % NF  == i
            Xtrain = X[~ival]
            Ytrain = Y[~ival]
            Xval   = X[ival]
            Yval   = Y[ival]
            checkpoint_path = self.checkpoint_path('m3D', fold=i, WIDTH=self.cfg.WIDTH, tag=self.cfg.tag)
            train(Xtrain, Ytrain, Xval, Yval, self.cfg, checkpoint_path=checkpoint_path)

if __name__ == '__main__':
    sys.exit(TrainNoduleApp().main() or 0)
