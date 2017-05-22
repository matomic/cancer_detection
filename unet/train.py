'''train'''
from __future__ import print_function
from __future__ import division

# stdlib
import datetime
import functools
import os
import pickle
import sys
import time
import argparse

# 3rd-party
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
from console import PipelineApp
from img_augmentation import ImageAugment
from load_data import ImgStream, TrainValidDataset
from models import get_unet
from utils import dice_coef, hist_summary, safejsondump

# DEBUGGING
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
            loss_func.__name__ = 'dice_coef'
        else:
            print("Cannot find associated config JSON: {}".format(chkpt_json))
        model = load_model(chkpt_path, custom_objects={
                #'optimizer' : getattr(opt, cfg.fitter['opt'])(**cfg.fitter['opt_arg']),
                'dice_coef' : loss_func, # name should match __name__
                })
        print("model loaded from {}".format(chkpt_path))
    else:
        model =  get_unet(cfg.unet.version, cfg.HEIGHT, cfg.WIDTH, **cfg.unet.params)
        model.compile(
                optimizer = getattr(opt, cfg.fitter['opt'])(**cfg.fitter['opt_arg']),
                loss = loss_func,
                )
        print("model loaded from scratch")
    set_trace()
    return model

class UnetTrainer(PipelineApp):
    '''Train Unet model with simple 2D slices'''
    def __init__(self):
        super(UnetTrainer, self).__init__()
        self.full_train_data = []

    def arg_parser(self):
        parser = super(UnetTrainer, self).arg_parser()
        parser = argparse.ArgumentParser(add_help=True,
                description='Train Unet',
                parents=[parser], conflict_handler='resolve')

        parser.add_argument('--debug', action='store_true',
                help='enter debug model')

        return parser

    def argparse_postparse(self, parsedArgs=None):
        super(UnetTrainer, self).argparse_postparse(parsedArgs)
        # Normally, img_mask_gen.py generates image/nodule mask pairs in self.cfg.root/img_mask, so that training data are reusable.  However,
        # sometimes we might want to experiment with different preprocessing approaches.  So here we expect training data to come from the same place
        # as where we will be saving the result.  To use "standard" training data, make a symbolic link from result_dir back to root dir.
        img_mask_dir = os.path.join(self.result_dir, 'img_mask')
        assert os.path.isdir(img_mask_dir), 'image/nodule mask pairs are expected in {}. It does not exist'.format(img_mask_dir)
        self.full_train_data = ImgStream(self.result_dir, "train", batch_size=self.cfg.fitter['batch_size'], unlabeled_ratio=self.cfg.unlabeled_ratio)
        set_trace()

    def _main_impl(self):
        if self.parsedArgs.debug:
            self.do_debug()
        else:
            self.do_train()

    def get_train_data(self, fold, augment_generator=None):
        '''Return training data generators for {fold}
        {augment_generator}, if not None, should be a function that takes a generator of training data and returns another generator of augmented input data.
        '''
        if fold < self.cfg.fitter['NCV']:# CV
            dataset = self.full_train_data.CV_fold_gen(fold, self.cfg.fitter['NCV'], shuffleTrain=True)
        else:#all training data
            dataset = self.full_train_data.all_gen(shuffle=True)

        if augment_generator:
            dataset = TrainValidDataset(
                    augment_generator(dataset.train),
                    dataset.validation,
                    dataset.size)
        return dataset

    def trainfor(self, model, tds, epochs, steps_per_epoch, chkpt_path):
        '''train {model} with training dataset {tds} for {epochs}'''
        batch_size = self.cfg.fitter['batch_size']
        steps_per_epoch = ((steps_per_epoch or 3000) // batch_size) * batch_size
        history = model.fit_generator(tds.train,
                steps_per_epoch  = steps_per_epoch,
                epochs           = epochs,
                validation_data  = tds.validation,
                validation_steps = tds.size // self.cfg.fitter['NCV'],
                callbacks        = fit_callbacks(chkpt_path)
                )
        return history

    checkpoint_path_fmrstr = '{net}_{WIDTH}_{tag}_fold{fold}.hdf5'
    def do_train(self):
        """
        build and train the CNNs.
        """
        np.random.seed(1234) # pylint: disable=no-member

        sess = K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        #image augmentation, flow
        imgAug = functools.partial(ImageAugment(**self.cfg.aug).flow_gen, mode='fullXY')

        folds_best_epoch = []
        folds_history = {}
        for fold in self.cfg.fitter['folds']:
            assert fold <= self.cfg.fitter['NCV']
            print("--- CV for fold: {}".format(fold))

            tds = self.get_train_data(fold, imgAug)
            if fold < self.cfg.fitter['NCV']:#CV
                num_epochs = self.cfg.fitter['num_epochs']
            else:
                assert False, 'bad branch'
                num_epochs = int(np.mean(folds_best_epoch)+1) if folds_best_epoch else self.cfg.fitter['num_epochs']

            #now_dt = datetime.datetime.now()
            #now_str = '.'.join(now_dt.isoformat().split('.')[:-1])
            #chkpt_path = os.path.join(self.cfg.params_dir, '{}_unet_{}_{}_fold{}.hdf5'.format(now_str, self.cfg.WIDTH, tag, fold))
            #chkpt_path = os.path.join(self.cfg.dirs.params_dir, self.parsedArgs.session, 'unet_{}_{}_fold{}.hdf5'.format(self.cfg.WIDTH, tag, fold))
            checkpoint_path = self.checkpoint_path('unet', fold, WIDTH=self.cfg.WIDTH, tag=self.cfg.tag)

            with K.tf.device('/gpu:0'):
                K.set_session(sess)
                model = model_factory(self.cfg, checkpoint_path)

            # Save configuration and model to JSON before training
            mjson_path = os.path.splitext(checkpoint_path)[0] + '.json'
            train_json = {
                    '__id__' : { 'tag' : self.cfg.tag, 'fold' : fold, 'ts': time.time(), 'dt' : datetime.datetime.now().isoformat() },
                    'config' : self.cfg.to_json(),
                    'model'  : model.get_config() # model architecture
                    }
            safejsondump(train_json, mjson_path, 'w')
#            set_trace()

            with K.tf.device('/gpu:0'):
                K.set_session(sess)
                history = self.trainfor(model, tds, num_epochs, steps_per_epoch=3000, chkpt_path=checkpoint_path)

#            set_trace()

            #history has epoch and history atributes
            train_json['history'] = folds_history[fold] = history.history
            folds_best_epoch.append(np.argmin(history.history['val_loss']))
            safejsondump(train_json, mjson_path)
            del model

        #save the full fitting history file for later study
        with open(os.path.join(self.cfg.dirs.log_dir,'history_{}_{}.pkl'.format(self.cfg.WIDTH, self.cfg.tag)),'wb') as f:
            pickle.dump(folds_history, f)
        #summary of all folds
        summary = hist_summary(folds_history)
        print(summary)

    def do_debug(self):
        '''run this when --debug is specified'''
        sess = K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        K.set_session(sess)
        chkpt_path = self.parsedArgs.chkpt_path or 'debug_checkpoint.h5'
        fold = 0
        model = model_factory(self.cfg, chkpt_path)
        model.summary()
        #a_func = functools.partial(ImageAugment(**config.aug).flow_gen, mode='fullXY')
        training_set = ImgStream(self.cfg.root, "train", batch_size=self.cfg.fitter['batch_size'], unlabeled_ratio=self.cfg.unlabeled_ratio)
        tds = training_set.all_gen(cycle=False, shuffle=False, test=False)
        loss = model.evaluate_generator(tds.train, steps=3000//self.cfg.fitter['batch_size'])
        print(loss)
        tds = self.get_train_data(training_set, fold)
        self.trainfor(model, tds, epochs=1, chkpt_path=chkpt_path,
                steps_per_epoch = 300 # quick epoch
                )
        assert os.path.isfile(chkpt_path)

if __name__=='__main__':
    sys.exit(UnetTrainer().main() or 0)

# eof
