# -*- coding: utf-8 -*-
'''₍⸍⸌̣ʷ̣̫⸍̣⸌₎'''
from __future__ import print_function, division

# stdlib
#import collections
import functools
#import glob
import itertools
import os
import pickle
import sys

# 3rd-party
from keras.models import load_model
from skimage import measure, segmentation
#from tqdm import tqdm
#import nibabel as nib
import numpy as np
import matplotlib.image as mplimage

# in-house
from console import PipelineApp
from img_mask_gen import LunaCase, get_lung_mask_npy, get_img_mask_npy
from utils import dice_coef
from train import model_factory

# DEBUGGING
#from pprint import pprint
#from ipdb import set_trace

npf32_t = np.float32 # pylint: disable=no-member

### loss fuction to make getmodel work, copied from segment/utils.py
def getmodel(tag, fold, cfg):
    '''Return the 2D Unet model of {tag} (version) with {fold}'''
    loss_func = functools.partial(dice_coef, smooth=10, pred_mul=1.0, p_ave=0.6, negate=True)
    model_path=os.path.join(cfg.params_dir, 'unet_{}_{}_fold{}.hdf5'.format(512, tag, fold))
    model = load_model(model_path, custom_objects={'loss': loss_func})
    return model

def normalize(x):
    '''Normalize from HU to [0,255] uint8 scale'''
    return np.interp(x, [-1000, 200], [0, 255], left=0, right=255).astype(np.uint8) # HU -_> [0, 255]

class NoduleDetectApp(PipelineApp):
    '''
    (つ・▽・)つ⊂(・▽・⊂)
    '''
    def __init__(self):
        super(NoduleDetectApp, self).__init__()
        self.output_dir = self.missed_dir = None

    def argparse_postparse(self, parsedArgs=None):
        super(NoduleDetectApp, self).argparse_postparse(parsedArgs)
        self.input_dir = os.path.join(self.cfg.root, 'img_mask')
        assert self.parsedArgs.session, 'unspecified unet training --session'
        self.output_dir = os.path.join(self.result_dir, 'nodule_candidates')
        self.missed_dir = os.path.join(self.result_dir, 'nodule_missed')
        self.provision_dirs(self.output_dir, self.missed_dir)

    def subset_input_dir(self, subset):
        '''{subset} data input'''
        return os.path.join(self.input_dir, 'subset{:d}'.format(subset))

    def lung_mask_path(self, subset, hashid):
        '''npy file path to save lung mask'''
        return os.path.join(self.subset_input_dir(subset), 'lungmask_{}.npy'.format(hashid))

    def ct_slice_path(self, subset, hashid, z):
        '''npy file path to save numpy array for slice {z} of patient {hashid} from {subset}'''
        return os.path.join(self.subset_input_dir(subset), 'image_{}_{:03d}.npy'.format(hashid, z))

    def nod_mask_path(self, subset, hashid, z):
        '''npy file path to save numpy array for label mask'''
        return os.path.join(self.subset_input_dir(subset), 'mask_{}_{:03d}.npy'.format(hashid, z))

    #def getmodel(self, fold):
    #    '''Return the 2D Unet model of {tag} (version) with {fold}'''
    #    loss_func = functools.partial(dice_coef, smooth=10, pred_mul=1.0, p_ave=0.6, negate=True)

    #    model_path=os.path.join(cfg.params_dir, 'unet_{}_{}_fold{}.hdf5'.format(512, tag, fold))
    #    model = load_model(model_path, custom_objects={'loss': loss_func})
    #    return model

    @staticmethod
    def predict_nodules(case, models, lung_mask, cut=0.3):
        '''Load image from {case}, normalize HU to int8, then to unit interval for input to {models}
        Cumulatively infere {nodules}: values at each voxel counts the number of models whose inferred nodule probability is > {cut}
        '''

        res = np.array(normalize(case.image)*lung_mask, dtype=npf32_t)*(1.0/255)-0.05
        res = np.expand_dims(res, axis=-1)
        nodules = models[0].predict(res,batch_size=8) > cut
        for model in models[1:]:
            nodules += model.predict(res,batch_size=8) > cut
        nodules = np.reshape(nodules, nodules.shape[:-1])
        return res, nodules

    def luna_nodule_detect(self, nodule_case_gen, models, output_shape):
        '''ʕ •̀ o •́ ʔ'''
        W, H = output_shape
        n = None
        for n, case in enumerate(nodule_case_gen):
            lmask_path = self.lung_mask_path(case.subset, case.hashid)
            lung_mask = get_lung_mask_npy(case, lmask_path, save_npy=False, lazy=True)
            ## Run inference from unet
            model_in, model_out = self.predict_nodules(case, models, lung_mask, cut=self.cfg.keep_prob)
            nodules = model_out * (lung_mask > 0.2)

            ## find nodules and its central location
            binary_nodule = np.array(nodules>=0.5, dtype=np.int8)
            labels = measure.label(binary_nodule) # label connected region of binary_nodule
            vals, counts = np.unique(labels, return_counts=True) # region label value and voxel count
            counts = counts[vals!=0]
            vals = vals[vals!=0]
            candidate_list = sorted(itertools.izip(counts, vals), reverse=True)

            ## keep only larger than some threshold volume
            h = case.spacing[2] / case.spacing[0]
            volume_threshold = 50
            nod_res =[]
            ls = [] # labels
            Ndeteced = 0
            detected_set = set()
            for count, val in candidate_list:
                ## Obvious candiate rejection
                if count <= volume_threshold:
                    continue # ignore region whose voxel count is not larger than some threshold

                xyz = np.where(labels == val)
                cz, cy, cx = tuple(int(np.median(x)) for x in xyz)
                #ss = tuple(int(np.max(x)-np.min(x))+1 for x in xyz)
                #if ss[0]<3: #thickness<3 pixels
                #    continue
                # output result image with WIDTH
                cx_min, cx_max = cx-W//2, cx+W//2
                cy_min, cy_max = cy-W//2, cy+W//2
                cz_min, cz_max = cz-H//2, cz+H//2

                if cy_min<=0 or cy_max>512 or cx_min<=0 or cx_max>512 or cz_min<=0 or cz_max>=model_in.shape[0]:
                    continue # ignore region that is outside of CT volume

                ## Add kept candidates to nod_res/ls list
                #try:
                out = np.transpose(model_in[cz_min:cz_max, cy_min:cy_max, cx_min:cx_max, 0]+0.05, (1,2,0)) # crop volumn, tranpose to y-x-z?
                #except Exception:
                #    continue
                assert out.shape == (W,W,H), 'bad shape: {!r} != {!r}'.format(out.shape, (W,W,H))

                nod_res.append(out)
                l = 0
                for xyzd in case.nodules:
                    x_nod, y_nod, z_nod, d_nod = xyzd
                    dz = np.abs(z_nod - cz)*h
                    dx = np.abs(x_nod - cx)
                    dy = np.abs(y_nod - cy)
                    if dx*dx+dy*dy+dz*dz <= d_nod**2/4:
                        l=1
                        if xyzd not in detected_set:
                            Ndeteced += 1
                            detected_set.add(xyzd)
                        else: # nodule detected in two disconnected regions
                            print('case {:03d}({}): nodule@(x{xyzd[0]:g},y{xyzd[1]:g},z{xyzd[2]:g},d{xyzd[3]:g}) detected again at (x{},y{},z{}).'.format(n, case.hashid, cx, cy, cz, xyzd=xyzd))
                        break
                ls.append(l)

            # save
            with open(os.path.join(self.output_dir,'{}.pkl'.format(case.hashid)), 'wb') as output:
                pickle.dump([nod_res, ls], output)

            # save missed detection
            for xyzd in case.nodules:
                if xyzd not in detected_set:
                    assert np.all(np.round(xyzd[:3]) == xyzd[:3]), 'bad xyz: {}'.format(xyzd[:3])
                    z_int = int(xyzd[2])
                    f = os.path.join(self.missed_dir, '{}-{}.png'.format(case.hashid, z_int))
                    img_path  = self.ct_slice_path(case.subset, case.hashid, z_int)
                    msk_path  = self.nod_mask_path(case.subset, case.hashid, z_int)
                    img, mask = get_img_mask_npy(case, lung_mask, z_int, img_path, msk_path, save_npy=False, lazy=True)
                    haslabel  = np.any(mask)
                    if not haslabel:
                        print('case {:03d}({}): z={} expected label.'.format(n, case.hashid, z_int))
                    mplimage.imsave(f, self.drawPrediction(img, mask, model_out[z_int,...]), vmin=0, vmax=255)
                    print('case {:03d}({}): nodule@(x{xyzd[0]:g},y{xyzd[1]:g},z{xyzd[2]:g},d{xyzd[3]:g}) missed -> {}'.format(n, case.hashid, f, xyzd=xyzd))
#                    set_trace()
            print('case {:03d}({}): {:d} nodules, {:2d} candidates, {:d} detected'.format(n, case.hashid, len(case.nodules), len(ls), Ndeteced))

        return n # number of cases

    @staticmethod
    def drawPrediction(image, expected, predicted):
        '''Compose {image} with {expected} and {predicted} boundaries'''
        vmin, vmax = image.min(), image.max()
        if image.ndim == 2:
            image = np.transpose([image, image, image], (1,2,0)) # color
        image[segmentation.find_boundaries(expected)] = [vmin, vmin, vmax]
        image[segmentation.find_boundaries(predicted)] = [vmax, vmin, vmin]
        return image

    checkpoint_path_fmrstr = '{net}_{WIDTH}_{tag}_fold{fold}.hdf5'
    def _main_impl(self):
        ## output nodule image size
        W, H = 64, 16

        #assert len(sys.argv)==2
        #tags = sys.argv[1].split(',')
        #cfg = __import__('config_v{}'.format(tags[0])) # NOTE: need to handle multiple tag case?
        #models = [getmodel(int(t), 0, cfg) for t in tags]

        checkpoint_path = self.checkpoint_path('unet', fold=0, WIDTH=self.cfg.WIDTH, tag=self.cfg.tag)
        assert os.path.isfile(checkpoint_path), checkpoint_path

        models = [model_factory(self.cfg, checkpoint_path)]

        df_node = LunaCase.readNodulesAnnotation(self.cfg.dirs.csv_dir)

        for subset in range(10):
            print("processing subset ",subset)
            subset_path = os.path.join(self.cfg.root, 'data', 'subset{}'.format(subset))

            case_gen = LunaCase.iterLunaCases(subset, subset_path, df_node, use_tqdm=False)
            self.luna_nodule_detect(case_gen, models, output_shape=(W, H))

if __name__ == '__main__':
    sys.exit(NoduleDetectApp().main() or 0)

# eof
