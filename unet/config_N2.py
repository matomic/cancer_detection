'''Configuration file'''
import os

from console import NetSpecs

tag = 2
#not working better
net = NetSpecs(
        name = 'Sample3D',
        version = 1,
        params = None
        )

#score
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
## ----- model specification ----
## -------------------------------------------------
#image size
WIDTH   = 64
HEIGHT  = 64
CHANNEL = 16

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

def main():
    '''write to JSON'''
    from utils import safejsondump
    config_js = os.path.splitext(__file__)[0] + '.json'
    with open(config_js, 'w') as fp:
        config_dict = {
                k:v for k,v in globals().iteritems()
                if not k.startswith('_')
                and isinstance(v, (int, float, basestring, dict, list, tuple, set))
                }
        safejsondump(config_dict, fp)

if __name__ == '__main__':
    main()

# eof
