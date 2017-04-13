from __future__ import division, print_function

import os.path
import sys
#import inspect

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import simplejson as json

from keras.models import model_from_json

import load_data as ld
#from img_augmentation import ImageAugment
#from train import makeModel

from pprint  import pprint
import ipdb

def main(cfg, prefix=''):
    """
    build and train the CNNs.
    """
    np.random.seed(1234)
    tag = cfg.tag

    #load data, flow
    #full_train_data = ld.ImgStream("train", cfg.fitter['batch_size'], unlabelled_ratio = cfg.unlabelled_ratio)

    #image augmentation, flow
    #imgAug = ImageAugment(**cfg.aug)

    #folds_best_epoch = []
    for fold in cfg.fitter['folds']:
#        assert fold <= cfg.fitter['NCV']
#        print("--- CV for fold: {}".format(fold))
#        if fold<cfg.fitter['NCV']:#CV
#            train_data, val_data  = full_train_data.CV_fold_gen(
#                    fold, cfg.fitter['NCV'], shuffleTrain=True)
#            #num_epochs = cfg.fitter['num_epochs']
#        else:#all training data
#            train_data = full_train_data.all_gen(shuffle=True)
#            #num_epochs = int(np.mean(folds_best_epoch)+1) if len(folds_best_epoch)>0 else cfg.fitter['num_epochs']
#            val_data = None

        imgs, labels, subsets = ld.get_file_list('train')
        subsets = np.asarray(subsets)
        n_imgs = len(imgs)
        batch_size = cfg.fitter['batch_size']
        n_batches = n_imgs // batch_size

        #model = makeModel(cfg)
        model_path = 'unet_{}_{}_fold{}.json'.format(cfg.WIDTH, tag, fold)
        if prefix:
            model_path = prefix + '_' + model_path
        model_path = os.path.join(cfg.params_dir, model_path)
        model_json = json.load(open(model_path, 'r'))
        if 'model_json' in model_json:
            json_str = model_json['model_json']
        else:
            json_str = model_json['model']
        if not isinstance(json_str, basestring):
            json_str = json.dumps(json_str)
        model = model_from_json(json_str)
        chkpt_path = os.path.splitext(model_path)[0] + '.hdf5'
        #model_json = os.path.splitext(checkpoint_path)[0] + '.json'
        #json.dump({
        #    'config' : { k:v for k,v in cfg.__dict__.iteritems() if not k.startswith('_') and not inspect.ismodule(v) },

        #    'model' : json.loads(model.to_json())
        #    },
        #    open(model_json, 'w'),
        #    indent=1,sort_keys=True,separators=(',',': '))
        model.load_weights(chkpt_path)
        ipdb.set_trace()
        bins = np.linspace(0, 1, 11)
        pos_hist = np.zeros(10, dtype=int)
        neg_hist = np.zeros(10, dtype=int)
        for b in xrange(n_batches):
            if b%100==0:
                print('batch {:03d}/{:03d}'.format(b, n_batches))
            batch_slice = slice(b*batch_size, (b+1)*batch_size)
            imgs_npy = np.asarray([np.load(x) for x in imgs[batch_slice]])
            pred_npy = model.predict(np.reshape(imgs_npy, imgs_npy.shape + (1,)))
            batch_label = np.asarray(labels[batch_slice])
            pos_hist += np.histogram(pred_npy[batch_label,...], bins=bins)[0]
            neg_hist += np.histogram(pred_npy[~batch_label,...], bins=bins)[0]
        pos_hist.astype(float)
        neg_hist.astype(float)
        chkpt_png = os.path.splitext(chkpt_path)[0] + '.png'
        plot_histgram(chkpt_png, pos_hist, neg_hist, model_json['config'])

def plot_histgram(chkpt_png, pos_hist, neg_hist, config_dic):
    '''Save PNG to {chkpt_png} a bar plot with {pos_hist} and {neg_hist}, generated from Unet {config_dic}
    '''
    left_npry = np.arange(0.1,1,0.1)
    width=0.05
    try:
        plt.figure(1)
        plt.clf()
        plt.bar(left_npry,       pos_hist[1:].astype(float)/pos_hist.sum(), align='edge', width=width)
        plt.bar(left_npry+width, neg_hist[1:].astype(float)/neg_hist.sum(), align='edge', width=width)
        plt.legend(['has label','unlabled'], loc='best')
        plt.title('unet_v{net_version}:aug={augment}:ulbl_ratio={unlabelled_ratio}'.format(**config_dic.get('config', config_dic)))
        plt.savefig(chkpt_png)
    except Exception as exc:
        raise
    finally:
        ipdb.set_trace()

if __name__ == '__main__':
    import config
    main(config, sys.argv[1])
