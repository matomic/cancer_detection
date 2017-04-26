from __future__ import print_function
from __future__ import division

import argparse
import os
import pickle
import sys

import pandas as pd
import numpy as np
# from keras import backend as K
from keras import callbacks
from keras import optimizers as opt

from img_augmentation2 import ImageDataGenerator
from models import get_3Dnet


def train(Xtrain, Ytrain, Xval, Yval, cfg, fold):
    #call backs
    model_checkpoint = callbacks.ModelCheckpoint(
            os.path.join(cfg.dirs.params_dir, 'm3D_{}_{}_fold{}.hdf5'.format(cfg.WIDTH, cfg.tag, fold)),
            monitor='val_loss', verbose=0,
            save_best_only=False
            )
    learn_rate_decay = callbacks.ReduceLROnPlateau(
            monitor='val_loss',factor=0.3,
            patience=8, min_lr=1e-5, verbose=1
            )
    earlystop = callbacks.EarlyStopping(monitor='val_loss',patience=15,mode='min',verbose=1)

    ## build the neural net work
    model = get_3Dnet(cfg.net.name, cfg.net.version, cfg.WIDTH, cfg.HEIGHT, cfg.CHANNEL)
    model.compile(
            optimizer= getattr(opt, cfg.fitter['opt'])(**cfg.fitter['opt_arg']),
            loss='binary_crossentropy'
            )
    datagen = ImageDataGenerator(**cfg.aug)
    datagenOne = ImageDataGenerator()

    #Fit here
    history = model.fit_generator(datagen.flow(Xtrain,Ytrain,batch_size=cfg.fitter['batch_size']),
            steps_per_epoch=len(Xtrain)//cfg.fitter['batch_size'],
            epochs=cfg.fitter['num_epochs'],
            validation_data = datagenOne.flow(Xval,Yval,batch_size=cfg.fitter['batch_size']),
            validation_steps = len(Xval)//cfg.fitter['batch_size'],
            callbacks = [learn_rate_decay, model_checkpoint, earlystop]
            )
    del model

def main(sys_argv=None):
    from console import parse_args
    parsedArgs = parse_args(sys_argv)

    np.random.seed(123)
    cfg = __import__(parsedArgs.config)
    nods_dir = os.path.join(cfg.root, 'nodule_candidate_{}/'.format('_'.join(parsedArgs.tag)))
    assert os.path.isdir(nods_dir), 'not a valid dir: {}'.format(nods_dir)
    print("loading {} ...".format(nods_dir))
    ids = os.listdir(nods_dir)
    data = [pickle.load(open(os.path.join(nods_dir, i),'rb')) for i in ids]
    Ncase = np.sum([len(x[0]) for x in data])

    Y = np.zeros(Ncase)
    W = cfg.WIDTH
    NC = cfg.CHANNEL
    X = np.zeros((len(Y), W, W, NC), dtype=np.float32); #set to 0 for empty chanels
    c = 0
    for x in data:
        nx = len(x[0])
        if nx==0:
            continue
        imgs = x[0]
        ls = x[1]
        Y[c:c+nx] = ls
        X[c:c+nx] = imgs
        c += nx

    print("total training cases ", Ncase)
    print("percent Nodules: ",np.sum(Y)/Ncase)

    # N fold cross validation
    NF = cfg.fitter['NCV']
    ii =  np.arange(Ncase)
    for i in cfg.fitter['folds']:
        ival = (ii%NF==i)
        Xtrain= X[~ival]
        Ytrain = Y[~ival]
        Xval = X[ival]
        Yval = Y[ival]
        train(Xtrain,Ytrain,Xval,Yval, cfg, i)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:] or None))
