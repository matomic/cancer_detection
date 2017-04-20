'''train'''
from __future__ import print_function, division

import collections
import datetime
import functools
import os
import pickle
import sys
import time

import simplejson as json
import numpy as np
#import pandas as pd
#import tensorflow as tf

#from keras import backend as K
import keras.backend.tensorflow_backend as K
from keras import callbacks
from keras import optimizers as opt
from keras.models import load_model
#from keras.metrics import categorical_accuracy
#from keras.objectives import categorical_crossentropy

from Unet import get_unet
from img_augmentation import ImageAugment
from load_data import ImgStream
import utils
import config

from pprint import pprint
import ipdb

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

def makeModel(cfg, chkpt_path=None):
    '''Build the neural network from scratch OR if {chkpt_path} is provided, load from checkpoint.'''
    loss_func = getattr(utils, cfg.loss_func)(**cfg.loss_args)
    ipdb.set_trace()
    if os.path.isfile(chkpt_path):
        model = load_model(chkpt_path, custom_objects={
                #'optimizer' : getattr(opt, cfg.fitter['opt'])(**cfg.fitter['opt_arg']),
                'loss' : loss_func,
                })
    else:
        model =  get_unet(cfg.HEIGHT, cfg.WIDTH, **cfg.unet)
        model.compile(
                optimizer = getattr(opt, cfg.fitter['opt'])(**cfg.fitter['opt_arg']),
                loss = loss_func,
                )
    return model

TrainingDataSet = collections.namedtuple('_TDS',
        ('full', 'train', 'validation'))

def get_train_data(full_train_data, fold, cfg, augment_generator=None):
    '''Return training data generators for {fold}
    {augment_generator}, if not None, should be a function that takes a generator of training data and returns another generator of augmented input data.
    '''
    if fold < cfg.fitter['NCV']:# CV
        train_data, val_data  = full_train_data.CV_fold_gen(fold, cfg.fitter['NCV'], shuffleTrain=True)
    else:#all training data
        train_data = full_train_data.all_gen(shuffle=True)
        val_data = None

    if augment_generator:
        train_data = augment_generator(train_data)
    return TrainingDataSet(full_train_data, train_data, val_data)

def trainfor(model, tds, epochs, chkpt_path, cfg, steps_per_epoch=None):
    '''train {model} with training dataset {tds} for {epochs}, saving to checkpoint {chkpt_path}'''
    batch_size = cfg.fitter['batch_size']
    steps_per_epoch = ((steps_per_epoch or 3000) // batch_size) * batch_size
    with K.tf.device('/gpu:0'):
        K.set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)))
        #variables = K.tf.global_variables()
        #K.get_session().run(K.tf.variables_initializer(variables))
        history = model.fit_generator(tds.train,
                steps_per_epoch  = steps_per_epoch,
                epochs           = epochs,
                validation_data  = tds.validation,
                validation_steps = len(tds.full) // cfg.fitter['NCV'],
                callbacks        = makeFittingCallbacks(chkpt_path)
                )
    return history

def main(cfg):
    """
    build and train the CNNs.
    """
    # Generate config dictionaries
    tag = cfg.tag

    np.random.seed(1234) # pylint: disable=no-member

    #load data, flow
    full_train_data = ImgStream("train", cfg.fitter['batch_size'], unlabeled_ratio = cfg.unlabeled_ratio)

    #image augmentation, flow
    imgAug = functools.partial(ImageAugment(**cfg.aug_args).flow_gen, mode='fullXY')

    folds_best_epoch = []
    folds_history = {}
    for fold in cfg.fitter['folds']:
        assert fold <= cfg.fitter['NCV']
        print("--- CV for fold: {}".format(fold))

        tds = get_train_data(full_train_data, fold, cfg, imgAug)
        if fold < cfg.fitter['NCV']:#CV
            num_epochs = cfg.fitter['num_epochs']
        else:
            assert False, 'bad branch'
            num_epochs = int(np.mean(folds_best_epoch)+1) if folds_best_epoch else cfg.fitter['num_epochs']

        #now_dt = datetime.datetime.now()
        #now_str = '.'.join(now_dt.isoformat().split('.')[:-1])
        #chkpt_path = os.path.join(cfg.params_dir, '{}_unet_{}_{}_fold{}.hdf5'.format(now_str, cfg.WIDTH, tag, fold))
        chkpt_path = os.path.join(cfg.params_dir, 'unet_{}_{}_fold{}.hdf5'.format(cfg.WIDTH, tag, fold))
        with K.tf.device('/gpu:0'):
            K.set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)))
            model = makeModel(cfg, chkpt_path)

        # Save configuration and model to JSON before training
        mjson_path = os.path.splitext(chkpt_path)[0] + '.json'
        train_json = {
                '__id__' : { 'tag' : cfg.tag, 'fold' : fold, 'ts': time.time(), 'dt' : datetime.datetime.now().isoformat() },
                'config' : cfg.to_json(),
                'model' : json.loads(model.to_json()) # load using keras.models.model_from_json
                }
        utils.safejsondump(train_json, open(mjson_path, 'w'))

        history = trainfor(model, tds, num_epochs, chkpt_path, cfg, steps_per_epoch=3000)

        #history has epoch and history atributes
        train_json['history'] = folds_history[fold] = history.history
        folds_best_epoch.append(np.argmin(history.history['val_loss']))
        utils.safejsondump(train_json, open(mjson_path, 'w'))
        ipdb.set_trace()
        del model

    #save the full fitting history file for later study
    with open(os.path.join(cfg.log_dir,'history_{}_{}.pkl'.format(cfg.WIDTH, tag)),'wb') as f:
        pickle.dump(folds_history, f)
    #summary of all folds
    summary = utils.hist_summary(folds_history)
    print(summary)

def _debug_main(cfg):
    chkpt_path = 'debug_checkpoint.h5'
    fold = 0
    model = makeModel(config, chkpt_path)
    model.summary()
    #a_func = functools.partial(ImageAugment(**config.aug).flow_gen, mode='fullXY')
    tds = get_train_data(
            ImgStream("train", cfg.fitter['batch_size'], unlabeled_ratio = cfg.unlabeled_ratio),
            fold,
            cfg)
    ipdb.set_trace()
    trainfor(
            model      = model,
            tds        = tds,
            epochs     = 1,
            chkpt_path = chkpt_path,
            cfg        = cfg,
            steps_per_epoch = 300 # quick epoch
            )

if __name__=='__main__':
    if '--debug' in sys.argv:
        _debug_main(config)
    else:
        main(config)

# eof
