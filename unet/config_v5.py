'''Configuration file'''
import os
#score - 0.75
tag = 5

## UNet architecture
unet = {
        'net_version' : 3,
        'downsample_conv_repeat' : 2, # number of repeat for each convolution layer during down sample
        'upsample_conv_repeat' : 1,   # number of repeats for each convolution layer during upsample
        }

## -------------------------------------------------
## ---- data files ----
## -------------------------------------------------
root = os.environ.setdefault('LUNA_DIR', '/home/qiliu/Share/Clinical/lung/luna16/')

#make these directories
cache_dir  = os.path.join(root, 'cache')
params_dir = os.path.join(root, 'params')
res_dir    = os.path.join(root, 'results')
log_dir    = os.path.join(root, 'log')
csv_dir    = os.path.join(root, 'CSVFILES')
for d in [cache_dir,params_dir, res_dir, log_dir, csv_dir]:
    if not os.path.isdir(d):
        os.makedirs(d)

## -------------------------------------------------
## -----  fitter -------
## -------------------------------------------------
#sample
unlabeled_ratio = 0.5
fitter = {
    'batch_size' : 8,
    'num_epochs' : 40,
    'NCV'        : 5, # nonlinear cross validation
    'folds'      : [0], #fold==NCV means to use all data
    'opt'        : 'Adam',
    'opt_arg'    : { 'lr' : 1e-3 }
}

#loss function paramters
loss = {
        'smooth'   : 5,
        'pred_mul' : 0.6,
        'p_ave'    : 0.8
}

## -------------------------------------------------
## ----- model specification ----
## -------------------------------------------------
inp = { # input dimension
        'WIDTH'   : 512,
        'HEIGHT'  : 512,
        'CHANNEL' : 1
        }

## -------------------------------------------------
## ----- image augmentation parameters ----
## -------------------------------------------------
aug = {
        'featurewise_center'            : False,
        'samplewise_center'             : False,
        'featurewise_std_normalization' : False,
        'samplewise_std_normalization'  : False,
        'zca_whitening'                 : False,
        'rotation_range'                : 10.0,
        'width_shift_range'             : 0.20,
        'height_shift_range'            : 0.20,
        'shear_range'                   : 0.05,
        'zoom_range'                    : 0.1,
        'channel_shift_range'           : 0.,
        'fill_mode'                     : 'constant',
        'cval'                          : 0.,
        'horizontal_flip'               : True,
        'vertical_flip'                 : False,
        'rescale'                       : None,
}

# test_aug = None
use_augment = True
