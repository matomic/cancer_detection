'''train'''
from __future__ import print_function
from __future__ import division

# stdlib
import datetime
import functools
import os
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

    return [
            learn_rate_decay,
            model_checkpoint,
            earlystop
            ]

def dice_loss(loss_args):
    '''dice coefficient loss function'''
    loss_func = functools.partial(dice_coef, negate=True, **loss_args)
    loss_func.__name__ = 'dice_coef'
    return loss_func

def unet_from_checkpoint(chkpt_path, loss_args):
    '''Load pre-trained Unet model from {chkpt_path}'''
    #if cfg is None:
    #    chkpt_json = os.path.splitext(chkpt_path)[0] + '.json'
    #    if os.path.isfile(chkpt_json):
    #        # load loss function from accompanied configuration JSON
    #        cfg = json.load(open(chkpt_json, 'r'))['config']
    #        loss_args = cfg['loss_args']
    #    else:
    #        print("Cannot find associated config JSON: {}".format(chkpt_json))
    #else:
    #    loss_args = cfg.loss_args
    #loss_func = functools.partial(dice_coef, negate=True, **loss_args)
    #loss_func.__name__ = 'dice_coef' # Keras explicitly check loss function's __name__ even if customized
    model = load_model(chkpt_path, custom_objects={
        'dice_coef' : dice_loss(loss_args)
        })
    print("model loaded from {}".format(chkpt_path))
    return model

def unet_from_scratch(cfg, loss_args):
    '''Load untrained Unet model, with configuration {cfg}'''
    model =  get_unet(cfg.unet.version, cfg.HEIGHT, cfg.WIDTH, **cfg.unet.params)
    model.compile(
            optimizer = getattr(opt, cfg.fitter['opt'])(**cfg.fitter['opt_arg']),
            loss = dice_loss(loss_args),
            )
    print("model loaded from scratch")
    return model


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

        parser.add_argument('--debug', action='store_true',
                help='enter debug model')

        return parser

    def argparse_postparse(self, parsedArgs=None):
        super(UnetTrainer, self).argparse_postparse(parsedArgs)
        self.fitter = self.unet_cfg.fitter
        # Normally, img_mask_gen.py generates image/nodule mask pairs in self.cfg.root/img_mask, so that training data are reusable.  However,
        # sometimes we might want to experiment with different preprocessing approaches.  So here we expect training data to come from the same place
        # as where we will be saving the result.  To use "standard" training data, make a symbolic link from result_dir back to root dir.
        img_mask_dir = os.path.join(self.result_dir, 'img_mask')
        assert os.path.isdir(img_mask_dir), 'image/nodule mask pairs are expected in {}. It does not exist'.format(img_mask_dir)
        self.full_train_data = ImgStream(self.result_dir, "train", batch_size=self.fitter['batch_size'], unlabeled_ratio=self.unet_cfg.unlabeled_ratio)

    def _main_impl(self):
        if self.parsedArgs.debug:
            self.do_debug()
        else:
            self.do_train()

    def get_train_data(self, fold, NCV, augment_generator=None):
        '''Return training data generators for {fold}
        {augment_generator}, if not None, should be a function that takes a generator of training data and returns another generator of augmented input data.
        '''
        if fold < NCV:# CV
            dataset = self.full_train_data.CV_fold_gen(fold, NCV, shuffleTrain=True)
        else:#all training data
            dataset = self.full_train_data.all_gen(shuffle=True)

        if augment_generator:
            dataset = TrainValidDataset(
                    augment_generator(dataset.train),
                    dataset.validation,
                    dataset.size)
        return dataset

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

            tds = self.get_train_data(fold, NCV, imgAug)
            if fold < NCV:#CV
                num_epochs = self.fitter['num_epochs']
            else:
                assert False, 'bad branch'
                num_epochs = int(np.mean(folds_best_epoch)+1) if folds_best_epoch else self.fitter['num_epochs']

            #now_dt = datetime.datetime.now()
            #now_str = '.'.join(now_dt.isoformat().split('.')[:-1])
            #chkpt_path = os.path.join(self.cfg.params_dir, '{}_unet_{}_{}_fold{}.hdf5'.format(now_str, self.cfg.WIDTH, tag, fold))
            #chkpt_path = os.path.join(self.cfg.dirs.params_dir, self.parsedArgs.session, 'unet_{}_{}_fold{}.hdf5'.format(self.cfg.WIDTH, tag, fold))
            checkpoint_path = self.checkpoint_path('unet', fold, WIDTH=self.unet_cfg.WIDTH, tag=self.unet_cfg.tag)
            configjson_path = self.subst_ext(checkpoint_path, '.json')

            with K.tf.device('/gpu:0'):
                K.set_session(sess)
                if os.path.isfile(checkpoint_path):
                    if os.path.isfile(configjson_path):
                        loss_args = json.load(configjson_path)['config']['loss_args']
                    else:
                        loss_args = self.unet_cfg.loss_args
                        print("Cannot find associated config JSON: {}".format(configjson_path))
                    model = unet_from_checkpoint(checkpoint_path, **loss_args)
                else:
                    model = unet_from_scratch(self.unet_cfg, self.unet_cfg.loss_args)
                model.summary()

            # Save configuration and model to JSON before training
            train_json = {
                    '__id__'  : { 'name' : 'unet', 'tag' : self.unet_cfg.tag, 'fold' : fold, 'ts': time.time(), 'dt' : datetime.datetime.now().isoformat() },
                    'config'  : self.unet_cfg.to_json(),
                    #'model'   : model.get_config(), # model architecture
                    'model'   : { 'checkpoint_path' : checkpoint_path },
                    #'history' : folds_history,
                    }
            safejsondump(train_json, configjson_path, 'w')

            with K.tf.device('/gpu:0'):
                K.set_session(sess)
                set_trace()
                num_epochs = 1
                epoch_size = 3000
                history = model.fit_generator(tds.train,
                        steps_per_epoch  = (epoch_size // batch_size) * batch_size,
                        epochs           = num_epochs,
                        validation_data  = tds.validation,
                        validation_steps = tds.size // NCV,
                        callbacks        = fit_callbacks(checkpoint_path)
                        )

            #history has epoch and history atributes
            train_json['history'] = folds_history[fold] = history.history
            folds_best_epoch.append(np.argmin(history.history['val_loss']))
            safejsondump(train_json, configjson_path)
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
        chkpt_path = self.parsedArgs.chkpt_path or 'debug_checkpoint.h5'
        fitter = self.unet_cfg.fitter
        fold = 0
        if os.path.isfile(chkpt_path):
            model = unet_from_checkpoint(chkpt_path, self.unet_cfg.loss_args)
        else:
            model = unet_from_scratch(self.unet_cfg, self.unet_cfg.loss_args)
        model.summary()
        #a_func = functools.partial(ImageAugment(**config.aug).flow_gen, mode='fullXY')
        training_set = ImgStream(self.root, "train", batch_size=fitter['batch_size'], unlabeled_ratio=self.unet_cfg.unlabeled_ratio)
        tds = training_set.all_gen(cycle=False, shuffle=False, test=False)
        loss = model.evaluate_generator(tds.train, steps=3000//fitter['batch_size'])
        print(loss)
        tds = self.get_train_data(training_set, fold)
        #self.trainfor(model, tds, epochs=1, chkpt_path=chkpt_path,
        #        steps_per_epoch = 300 # quick epoch
        #        )
        assert os.path.isfile(chkpt_path)

if __name__=='__main__':
    sys.exit(UnetTrainer().main() or 0)

# eof
