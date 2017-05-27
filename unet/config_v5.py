'''Configuration file'''
from console import NetSpecs

#score - 0.75
tag = 5

## UNet architecture
unet = NetSpecs(
        name    = 'unet',
        version = 3,
        params  = {
            'subsampling_conv_repeat' : 2,
            'upsampling_conv_repeat'  : 1,
            })

keep_prob=0.3
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
loss_func = 'dice_coef_loss_gen'
loss_args = {
        'smooth'   : 5,
        'pred_mul' : 0.6,
        'p_ave'    : 0.8
}

## -------------------------------------------------
## ----- model specification ----
## -------------------------------------------------
WIDTH   = 512
HEIGHT  = 512
CHANNEL = 1

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

# eof
