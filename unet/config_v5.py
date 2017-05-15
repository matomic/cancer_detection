'''Configuration file'''
import os

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

## -------------------------------------------------
## ---- data files ----
## -------------------------------------------------
root = os.environ.setdefault('LUNA_DIR', '/home/qiliu/Share/Clinical/lung/luna16/')

#make these directories
dirs = {
        'data_dir'   : os.path.join(root, 'data'),
        'cache_dir'  : os.path.join(root, 'cache'),
        'params_dir' : os.path.join(root, 'params'),
        'res_dir'    : os.path.join(root, 'results'),
        'log_dir'    : os.path.join(root, 'log'),
        'csv_dir'    : os.path.join(root, 'CSVFILES'),
        }

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

def to_json():
    return {
            k:v for k,v in globals().iteritems()
            if not k.startswith('_')
            and isinstance(v, (int, float, basestring, dict, list, tuple, set))
            }
def main():
    '''write to JSON'''
    from utils import safejsondump
    config_js = os.path.splitext(__file__)[0] + '.json'
    with open(config_js, 'w') as fp:
        safejsondump(to_json(), fp)

if __name__ == '__main__':
    main()

# eof
