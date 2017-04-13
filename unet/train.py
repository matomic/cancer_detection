#!/usr/bin/python3
'''train'''
from __future__ import print_function

#import datetime
import inspect
import os
import pickle
#import sys
#import time

import simplejson as json
import numpy as np
#import pandas as pd

#from keras import backend as K
from keras import callbacks
from keras import optimizers as opt
#from keras.metrics import categorical_accuracy
#from keras.objectives import categorical_crossentropy

from img_augmentation import ImageAugment
import Unet
import config
import load_data as ld
import utils

from pprint import pprint
import ipdb

class dotdict(dict):
    """ A dictionary whose attributes are accessible by dot notation.
    This is a variation on the classic `Bunch` recipe (which is more limited
    and doesn't give you all of dict's methods). It is just like a dictionary,
    but its attributes are accessible by dot notation in addition to regular
    `dict['attribute']` notation. It also has all of dict's methods.
    .. doctest::
        >>> dd = dotdict(foo="foofoofoo", bar="barbarbar")
        >>> dd.foo
        'foofoofoo'
        >>> dd.foo == dd['foo']
        True
        >>> dd.bar
        'barbarbar'
        >>> dd.baz
        >>> dd.qux = 'quxquxqux'
        >>> dd.qux
        'quxquxqux'
        >>> dd['qux']
        'quxquxqux'
    NOTE:   There are a few limitations, but they're easy to avoid
            (these should be familiar to JavaScript users):
        1.  Avoid trying to set attributes whose names are dictionary methods,
            for example 'keys'. Things will get clobbered. No good.
        2.  You can store an item in a dictionary with the key '1' (`foo['1']`),
            but you will not be able to access that attribute using dotted
            notation if its key is not a valid Python variable/attribute name
            (`foo.1` is not valid Python syntax).
    FOR MORE INFORMATION ON HOW THIS WORKS, SEE:
    - http://stackoverflow.com/questions/224026/
    - http://stackoverflow.com/questions/35988/
    """
    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        else:
            raise AttributeError(attr)

    __setattr__ = dict.__setitem__

    __delattr__ = dict.__delitem__

    @classmethod
    def recursiveFrom(cls, d):
        '''Generate instance recursively from dict {d}'''
        return cls({
            k : cls.recursiveFrom(v) if isinstance(v, dict) else v
            for k, v in d.iteritems()})

def makeFittingCallbacks(chkpt_path):
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
    #return [learn_rate_decay, model_checkpoint, earlystop, tensorboard]
    return [learn_rate_decay, model_checkpoint, earlystop]

def makeModel(cfg):
    '''Build the neural network'''
    net_cfg = cfg.get('unet', cfg)
    model =  Unet.get_unet(cfg.inp.HEIGHT, cfg.inp.WIDTH, net_cfg.net_version, **net_cfg)
    loss_cfg = cfg.get('loss', cfg)
    model.compile(
            optimizer= getattr(opt, cfg.fitter['opt'])(**cfg.fitter['opt_arg']),
            loss=utils.dice_coef_loss_gen(**loss_cfg)
            )
    return model

def main(cfg):
    """
    build and train the CNNs.
    """
    cfg = { k:v for k,v in cfg.__dict__.iteritems() if not k.startswith('_') and not inspect.ismodule(v) }
    cfg = dotdict.recursiveFrom(cfg)

    np.random.seed(1234)
    tag = cfg.tag

    #load data, flow
    full_train_data = ld.ImgStream("train", cfg.fitter['batch_size'], unlabeled_ratio = cfg.unlabeled_ratio)

    #image augmentation, flow
    imgAug = ImageAugment(**cfg.aug)

    folds_best_epoch = []
    folds_history = {}
    for fold in cfg.fitter['folds']:
        assert fold <= cfg.fitter['NCV']
        print("--- CV for fold: {}".format(fold))
        if fold<cfg.fitter['NCV']:#CV
            train_data, val_data  = full_train_data.CV_fold_gen(
                    fold, cfg.fitter['NCV'], shuffleTrain=True)
            num_epochs = cfg.fitter['num_epochs']
        else:#all training data
            assert False, 'bad branch'
            train_data = full_train_data.all_gen(shuffle=True)
            num_epochs = int(np.mean(folds_best_epoch)+1) if len(folds_best_epoch)>0 else cfg.fitter['num_epochs']
            val_data = None

        if cfg.use_augment:
            train_data_aug = imgAug.flow_gen(train_data, mode='fullXY')
        else:
            train_data_aug = train_data
        model = makeModel(cfg)

        #Fit here
        #now_dt = datetime.datetime.now()
        #now_str = '.'.join(now_dt.isoformat().split('.')[:-1])
        #chkpt_path = os.path.join(cfg.params_dir, '{}_unet_{}_{}_fold{}.hdf5'.format(now_str, cfg.inp.WIDTH, tag, fold))
        chkpt_path = os.path.join(cfg.params_dir, 'unet_{}_{}_fold{}.hdf5'.format(cfg.inp.WIDTH, tag, fold))
        model_json = os.path.splitext(chkpt_path)[0] + '.json'
        json.dump({
            'config' : cfg,
            'model' : json.loads(model.to_json()) # load using keras.models.model_from_json
            },
            open(model_json, 'w'),
            indent=1,sort_keys=True,separators=(',',': '))
        samples_per_epoch = (3000//cfg.fitter['batch_size'])*cfg.fitter['batch_size']
        ipdb.set_trace()
        history = model.fit_generator(train_data_aug,
                #samples_per_epoch=len(full_train_data)*(cfg.fitter['NCV']-1)/cfg.fitter['NCV'],
                steps_per_epoch  = samples_per_epoch,
                epochs           = num_epochs,
                validation_data  = val_data,
                validation_steps = len(full_train_data)//cfg.fitter['NCV'],
                callbacks = makeFittingCallbacks(chkpt_path)
                )
        #history has epoch and history atributes
        folds_history[fold] = history.history
        folds_best_epoch.append(np.argmin(history.history['val_loss']))
        del model

    #save the full fitting history file for later study
    with open(os.path.join(cfg.log_dir,'history_{}_{}.pkl'.format(cfg.inp.WIDTH, tag)),'wb') as f:
        pickle.dump(folds_history, f)
    #summary of all folds
    summary = utils.hist_summary(folds_history)
    print(summary)

if __name__=='__main__':
    main(config)
