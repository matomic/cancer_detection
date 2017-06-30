'''Configuration file'''
from config import NetSpecs

tag = 2
#not working better

## -------------------------------------------------
## ----- model specification ----
## -------------------------------------------------
#image size
net = NetSpecs(
        name = 'Sample3D',
        version = 1,
        params = None,
        WIDTH   = 64,
        HEIGHT  = 64,
        CHANNEL = 16,
        )

## -------------------------------------------------
## -----  fitter -------
## -------------------------------------------------
#sample
fitter = {
        'batch_size' : 16,
        'num_epochs' : 100,
        'NCV' : 5,
        'folds' : [1],
        'opt' : 'Adam',
        'opt_arg' : { 'lr' : 2e-4 },
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
        'rotation_range'                : 15.0,
        'width_shift_range'             : 0.10,
        'height_shift_range'            : 0.10,
        'shear_range'                   : 0.05,
        'zoom_range'                    : 0.10,
        'channel_shift_range'           : 0.,
        'fill_mode'                     : 'constant',
        'cval'                          : 0.,
        'horizontal_flip'               : True,
        'vertical_flip'                 : False,
        'rescale'                       : None,
        }

# eof
