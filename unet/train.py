'''train'''
from __future__ import print_function, division

# stdlib
import argparse
import datetime
import functools
import os
import pickle
import time

# third-party
import simplejson as json
import numpy as np
#import pandas as pd
#import tensorflow as tf

from keras import callbacks, optimizers as opt
from keras.backend import tensorflow_backend as K
from keras.models import load_model
#from keras.metrics import categorical_accuracy
#from keras.objectives import categorical_crossentropy

# in-house
from Unet import get_unet
from img_augmentation import ImageAugment
from load_data import ImgStream, TrainValidDataset
from utils import dice_coef, hist_summary, safejsondump

# debugging
from pprint import pprint
from ipdb import set_trace

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
    return [learn_rate_decay, model_checkpoint, earlystop]

def model_factory(cfg, chkpt_path=None):
    '''Build the neural network from scratch OR if {chkpt_path} is provided, load from checkpoint.'''
    loss_func = functools.partial(dice_coef, negate=True, **cfg.loss_args)
    loss_func.__name__ = 'dice_coef' # Keras explicitly check loss function's __name__ even if customized
    if os.path.isfile(chkpt_path):
        chkpt_json = os.path.splitext(chkpt_path)[0] + '.json'
        if os.path.isfile(chkpt_json):
            # load loss function from accompanied configuration JSON
            cfg = json.load(open(chkpt_json,'r'))['config']
            loss_func = functools.partial(dice_coef, negate=True, **cfg['loss_args'])
        else:
            print("Cannot find associated config JSON: {}".format(chkpt_json))
        model = load_model(chkpt_path, custom_objects={
                #'optimizer' : getattr(opt, cfg.fitter['opt'])(**cfg.fitter['opt_arg']),
                'loss' : loss_func,
                })
        print("model loaded from {}".format(chkpt_path))
    else:
        model =  get_unet(cfg.HEIGHT, cfg.WIDTH, **cfg.unet)
        model.compile(
                optimizer = getattr(opt, cfg.fitter['opt'])(**cfg.fitter['opt_arg']),
                loss = loss_func,
                )
        print("model loaded from scratch")
    return model

def get_train_data(full_train_data, fold, cfg, augment_generator=None):
    '''Return training data generators for {fold}
    {augment_generator}, if not None, should be a function that takes a generator of training data and returns another generator of augmented input data.
    '''
    if fold < cfg.fitter['NCV']:# CV
        dataset = full_train_data.CV_fold_gen(fold, cfg.fitter['NCV'], shuffleTrain=True)
    else:#all training data
        dataset = full_train_data.all_gen(shuffle=True)

    if augment_generator:
        dataset = TrainValidDataset(
                augment_generator(dataset.train),
                dataset.validation,
                dataset.size)
    return dataset

def trainfor(model, tds, epochs, chkpt_path, cfg, steps_per_epoch=None):
    '''train {model} with training dataset {tds} for {epochs}, saving to checkpoint {chkpt_path}'''
    batch_size = cfg.fitter['batch_size']
    steps_per_epoch = ((steps_per_epoch or 3000) // batch_size) * batch_size
    history = model.fit_generator(tds.train,
            steps_per_epoch  = steps_per_epoch,
            epochs           = epochs,
            validation_data  = tds.validation,
            validation_steps = tds.size // cfg.fitter['NCV'],
            callbacks        = fit_callbacks(chkpt_path)
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

    sess = K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    #image augmentation, flow
    imgAug = functools.partial(ImageAugment(**cfg.aug).flow_gen, mode='fullXY')

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
            K.set_session(sess)
            model = model_factory(cfg, chkpt_path)

        # Save configuration and model to JSON before training
        mjson_path = os.path.splitext(chkpt_path)[0] + '.json'
        train_json = {
                '__id__' : { 'tag' : cfg.tag, 'fold' : fold, 'ts': time.time(), 'dt' : datetime.datetime.now().isoformat() },
                'config' : cfg.to_json(),
                'model'  : model.get_config() # model architecture
                }
        safejsondump(train_json, open(mjson_path, 'w'))

        with K.tf.device('/gpu:0'):
            K.set_session(sess)
            history = trainfor(model, tds, num_epochs, chkpt_path, cfg, steps_per_epoch=3000)

        #history has epoch and history atributes
        train_json['history'] = folds_history[fold] = history.history
        folds_best_epoch.append(np.argmin(history.history['val_loss']))
        safejsondump(train_json, open(mjson_path, 'w'))
        del model

    #save the full fitting history file for later study
    with open(os.path.join(cfg.log_dir,'history_{}_{}.pkl'.format(cfg.WIDTH, tag)),'wb') as f:
        pickle.dump(folds_history, f)
    #summary of all folds
    summary = hist_summary(folds_history)
    print(summary)

def _debug_main(parsed, cfg):
    sess = K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    K.set_session(sess)
    chkpt_path = parsed.cp_path or 'debug_checkpoint.h5'
    fold = 0
    model = model_factory(cfg, chkpt_path)
    model.summary()
    #a_func = functools.partial(ImageAugment(**config.aug).flow_gen, mode='fullXY')
    training_set = ImgStream("train", cfg.fitter['batch_size'], unlabeled_ratio = cfg.unlabeled_ratio)
    tds = training_set.all_gen(cycle=False, shuffle=False, test=False)
    loss = model.evaluate_generator(tds.train, steps=3000//cfg.fitter['batch_size'])
    print(loss)
    tds = get_train_data(training_set, fold, cfg)
    trainfor(model, tds, epochs=1, chkpt_path=chkpt_path, cfg=cfg,
            steps_per_epoch = 300 # quick epoch
            )
    assert os.path.isfile(chkpt_path)

if __name__=='__main__':
    import config
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument('--debug', action='store_true',
            help='enter debug model')
    parser.add_argument('--cp-path', action='store',
            help='path to check point file'
            )
    parsed = parser.parse_args()
    if parsed.debug:
        _debug_main(parsed, config)
    else:
        main(config)

# eof
