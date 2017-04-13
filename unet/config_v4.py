import sys
import os
# -0.68
tag = 4
net_version = 2
## -------------------------------------------------
## ---- data files ----
## -------------------------------------------------
root="/home/qiliu/Share/Clinical/lung/luna16/"

#make these directories
cache_dir = os.path.join(root, 'cache')
params_dir = os.path.join(root, 'params')
res_dir = os.path.join(root, 'results')
log_dir = os.path.join(root, 'log');
for d in [cache_dir,params_dir, res_dir, log_dir]:
    if not os.path.isdir(d):
        os.makedirs(d)

## -------------------------------------------------
## -----  fitter -------
## -------------------------------------------------
#sample
unlabeled_ratio = 0.5;
fitter = dict(
    batch_size = 8,
    num_epochs = 40,
    NCV = 5,
    folds = [0], #fold == NCV means to use all data
    opt = 'Adam',
    opt_arg =dict(lr=1.0e-3),
);
#loss function paramters
smooth = 5;
pred_loss_mul = 0.6;
p_ave = 0.8;

#fitter = dict(
#    batch_size = 6,
#    num_epochs = 10,
#    NCV = 5,
#    folds = [0], #fold == NCV means to use all data
#    opt = 'SGD',
#    opt_arg =dict(lr=1.0e-2, momentum=0.90, decay=1.0e-6),
#);

## -------------------------------------------------
## ----- model specification ----
## -------------------------------------------------
#image size
WIDTH=512;
HEIGHT=512;
CHANNEL=1;

## -------------------------------------------------
## ----- image augmentation parameters ----
## -------------------------------------------------
aug = dict(
	 featurewise_center=False,
	 samplewise_center=False,
	 featurewise_std_normalization=False,
	 samplewise_std_normalization=False,
	 zca_whitening=False,
	 rotation_range=10.0,
	 width_shift_range=0.20,
	 height_shift_range=0.20,
	 shear_range=0.05,
	 zoom_range=0.1,
	 channel_shift_range=0.,
	 fill_mode='constant',
	 cval=0.,
	 horizontal_flip=True,
	 vertical_flip=False,
	 rescale=None,
);

test_aug = None;
