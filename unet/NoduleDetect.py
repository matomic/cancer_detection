# -*- coding: utf-8 -*-
'''₍⸍⸌̣ʷ̣̫⸍̣⸌₎'''
from __future__ import print_function, division

# stdlib
import argparse
import collections
#import functools
#import glob
#import itertools
import os
import pickle
import sys

# 3rd-party
#from keras.models import load_model
from skimage import measure, segmentation
#from tqdm import tqdm
#import nibabel as nib
import numpy as np
import matplotlib.image as mplimage

# in-house
from console import PipelineApp
from img_mask_gen import LunaCase, get_lung_mask_npy, get_img_mask_npy, normalize
from utils import safejsondump
from train import load_unet, UnetTrainer
from load_data import ImgStream

# DEBUGGING
from pprint import pprint
try:
    from ipdb import set_trace
except:
    pass

npf32_t = np.float32 # pylint: disable=no-member


NoduleCandidate = collections.namedtuple('NoduleCandidate', ('volume_array', 'x', 'y', 'z', 'd', 'is_nodule'))


class NoduleDetectApp(PipelineApp):
    '''
    (つ・▽・)つ⊂(・▽・⊂)
    '''
    def __init__(self):
        super(NoduleDetectApp, self).__init__()
        self.output_dir = self.missed_dir = None
        self.statistics = None

    def arg_parser(self):
        parser = super(NoduleDetectApp, self).arg_parser()
        parser = argparse.ArgumentParser(add_help=True,
                description='Generate nodule candidates from trained Unet',
                parents=[parser], conflict_handler='resolve')

        parser.add_argument('--no-lung-mask', action='store_true',
                help='If True, do not use lung segmentation.')

        parser.add_argument('--lazy', action='store_true',
                help='If True, use slice image, nodule mask and lung mask npy files saved in the output directory by previous session.')

        return parser

    def argparse_postparse(self, parsedArgs=None):
        super(NoduleDetectApp, self).argparse_postparse(parsedArgs)
        assert self.unet_cfg is not None, '--config-unet required.'
        assert self.n3d_cfg is not None, '--config-n3d required.'

        self.input_dir = os.path.join(self.result_dir, 'img_mask')
        assert self.parsedArgs.session, 'unspecified unet training --session'
        assert os.path.isdir(self.input_dir), 'data from needed directory not found: {}'.format(self.input_dir)
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
    def predict_nodules(case, models, lung_mask=None, cut=0.3, batch_size=8):
        '''Load image from {case}, normalize HU to int8, then to unit interval with estimated mean offset for input to {models}
        Cumulatively infere for nodule candidates: values at each voxel counts the number of models whose inferred nodule probability is > {cut}
        '''
        res = normalize(case.image)
        if lung_mask is not None:
            res *= lung_mask
        res = ImgStream.normalize_image(res)
        #res = np.array(res, dtype=npf32_t)*(1.0/255) - 0.05

        input_shape = {tuple(d.value for d in x.input.shape) for x in models}
        assert len(input_shape) == 1, 'Not-unique input shape: {}'.format(input_shape)
        input_shape = input_shape.pop()

        if len(input_shape) == 5:
            raise NotImplementedError
        elif len(input_shape) == 4:
            if input_shape[-1] == 1:
                res = np.expand_dims(res, axis=-1)
            elif input_shape[-1] == 3:
                Nz = res.shape[0]
                res = np.stack([
                    res[[0, *range(Nz-1)], ...],
                    res,
                    res[[*range(1,Nz), -1],...],
                    ], axis=-1)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        nodules = models[0].predict(res, batch_size=batch_size) > cut
        for model in models[1:]:
            nodules += model.predict(res, batch_size=batch_size) > cut
        nodules = np.reshape(nodules, nodules.shape[:-1])
        return res, nodules

    @staticmethod
    def detect_nodule_candidate(model_in, model_out, output_shape, lung_mask=None, volume_threshold=50, training_case=None):
        '''Generate list of nodules candidates and (if {training_case} is provided) set of matched nodules'''
        if lung_mask is None:
            nodules = model_out
        else:
            nodules = model_out * (lung_mask > 0.2)

        ## find nodules and its central location
        binary_nodule = np.array(nodules>=0.5, dtype=np.int8)
        labels = measure.label(binary_nodule) # label connected region of binary_nodule
        vals, counts = np.unique(labels, return_counts=True) # region label value and voxel count
        counts = counts[vals!=0]
        vals = vals[vals!=0]

        candidate_list = []
        detected_set = set()
        W, H, C = output_shape

        for count, val in sorted(zip(counts, vals), reverse=True):
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
            cy_min, cy_max = cy-H//2, cy+H//2
            cz_min, cz_max = cz-C//2, cz+C//2

            if cy_min<=0 or cy_max>512 or cx_min<=0 or cx_max>512 or cz_min<=0 or cz_max>=model_in.shape[0]:
                continue # ignore region that is outside of CT volume

            ## Slice input volume
            #try:
            out = np.transpose(model_in[cz_min:cz_max, cy_min:cy_max, cx_min:cx_max, 0]+0.05, (1,2,0)) # crop volumn, tranpose to y-x-z?
            #except Exception:
            #    continue
            assert out.shape == (W,H,C), 'bad shape: {!r} != {!r}'.format(out.shape, (W,H,C))

            ## Attach training label, 0 or 1, for whether this candidate is a known nodule. If training_case is None, label None.
            if training_case is None:
                candidate = NoduleCandidate(out, cx, cy, cz, 0, None) # FIXME
            else:
                isNodule = False
                for xyzd in training_case.nodules:
                    x_nod, y_nod, z_nod, d_nod = xyzd
                    dz = np.abs(z_nod - cz) * training_case.h
                    dx = np.abs(x_nod - cx)
                    dy = np.abs(y_nod - cy)
                    if dx*dx+dy*dy+dz*dz <= d_nod**2/4:
                        isNodule = True
                        if xyzd not in detected_set:
                            detected_set.add(xyzd)
                        else: # nodule detected in two disconnected regions
                            print('case {}: nodule@(x{xyzd[0]:g},y{xyzd[1]:g},z{xyzd[2]:g},d{xyzd[3]:g}) detected again at (x{},y{},z{}).'.format(training_case.hashid, cx, cy, cz, xyzd=xyzd))
                        break
                candidate = NoduleCandidate(out, cx, cy, cz, 0, isNodule)
            candidate_list.append(candidate)
        return candidate_list, detected_set

    def luna_nodule_detect(self, nodule_case_gen, models, output_shape):
        '''ʕ •̀ o •́ ʔ'''
        for n, case in enumerate(nodule_case_gen):
            output_path = os.path.join(self.output_dir,'{}.pkl'.format(case.hashid))
            if self.parsedArgs.lazy and os.path.isfile(output_path):
                print('case {:03d}({}): skipped'.format(n, case.hashid))
                continue

            case.readImageFile()
            if self.parsedArgs.no_lung_mask:
                lung_mask = None
            else:
                lmask_path = self.lung_mask_path(case.subset, case.hashid)
                assert os.path.isfile(lmask_path), 'lung mask file {} not found. Use --no-lung-mask?'.format(lmask_path)
                lung_mask = get_lung_mask_npy(case, lmask_path, save_npy=False, lazy=True)

            ## Run inference from unet
            model_in, model_out = self.predict_nodules(case, models, lung_mask, cut=self.unet_cfg.keep_prob)

            ## List of nodule candidates
            candidate_list, detected_set = self.detect_nodule_candidate(model_in, model_out, output_shape, lung_mask=lung_mask, training_case=case)

            # save candidate list
            with open(os.path.join(self.output_dir,'{}.pkl'.format(case.hashid)), 'wb') as output:
                pickle.dump([tuple(x) for x in candidate_list], output) # pickle does not work with namedtuple :(

            # save missed detection
            for xyzd in case.nodules:
                assert np.all(np.round(xyzd[:3]) == xyzd[:3]), 'bad xyz: {}'.format(xyzd[:3])
                if xyzd in detected_set:
                    #self.statistics['detected_nodule_list'].append(case.nodules)
                    self.statistics['nodules_detected'] += 1
                else:
                    z_int = int(xyzd[2])
                    f = os.path.join(self.missed_dir, '{}-{}.png'.format(case.hashid, z_int))
                    img_path  = self.ct_slice_path(case.subset, case.hashid, z_int)
                    msk_path  = self.nod_mask_path(case.subset, case.hashid, z_int)
                    img, mask = get_img_mask_npy(case, lung_mask, z_int, img_path, msk_path, save_npy=False, lazy=True)
                    haslabel  = np.any(mask)
                    if not haslabel:
                        print('case {:03d}({}): z={} expected label.'.format(n, case.hashid, z_int))
                    if mask is None:
                        mask = np.zeros_like(img, dtype=bool)
                    mplimage.imsave(f, self.drawPrediction(img, mask, model_out[z_int,...]), vmin=0, vmax=255)
                    print('case {:03d}({}): nodule@(x{xyzd[0]:g},y{xyzd[1]:g},z{xyzd[2]:g},d{xyzd[3]:g}) missed -> {}'.format(n, case.hashid, f, xyzd=xyzd))
                    #self.statistics['missed_nodule_list'].append(case.nodules)
                    self.statistics['nodules_missed'] += 1
            n_candids  = len(candidate_list)
            n_nodules  = len(case.nodules)
            n_detected = len(detected_set)
            self.statistics['n_cases'] += 1
            self.statistics['candidates'] += n_candids
            self.statistics['nodules'] += n_nodules
            if n_nodules > 0 and n_detected == 0: # failed to detect any nodule
                self.statistics.setdefault('case_missed', []).append(case.suid)
            print('case {:03d}({}): {:d} nodules, {:2d} candidates, {:d} detected'.format(n, case.hashid, n_nodules, n_candids, len(detected_set)))

    @staticmethod
    def drawPrediction(image, expected, predicted):
        '''Compose {image} with {expected} and {predicted} boundaries'''
        vmin, vmax = image.min(), image.max()
        if image.ndim == 2:
            image = np.transpose([image, image, image], (1,2,0)) # color
        image[segmentation.find_boundaries(expected)] = [vmin, vmin, vmax]
        image[segmentation.find_boundaries(predicted)] = [vmax, vmin, vmin]
        return image

    checkpoint_path_fmrstr = UnetTrainer.checkpoint_path_fmrstr
    def _main_impl(self):
        ## output nodule image size
        output_shape = (self.n3d_cfg.net.WIDTH, self.n3d_cfg.net.HEIGHT, self.n3d_cfg.net.CHANNEL)

        self.statistics = collections.defaultdict(int)
        #self.statistics['detected_nodule_list'] = []
        #self.statistics['missed_nodule_list'] = []

        #assert len(sys.argv)==2
        #tags = sys.argv[1].split(',')
        #cfg = __import__('config_v{}'.format(tags[0])) # NOTE: need to handle multiple tag case?
        #models = [getmodel(int(t), 0, cfg) for t in tags]

        checkpoint_path = self.checkpoint_path('unet', fold=0, WIDTH=self.unet_cfg.net.WIDTH, tag=self.unet_cfg.tag)
        assert os.path.isfile(checkpoint_path), checkpoint_path

        models = [load_unet(self.unet_cfg, checkpoint_path=checkpoint_path)]

        df_node = LunaCase.readNodulesAnnotation(self.dirs.csv_dir)

        for subset in range(10):
            if self.parsedArgs.subset and subset not in self.parsedArgs.subset:
                continue
            print("processing subset ",subset)
            case_gen = LunaCase.iterLunaCases(self.dirs.data_dir, subset, df_node, use_tqdm=False)
            self.luna_nodule_detect(case_gen, models, output_shape=output_shape)
        if not self.parsedArgs.lazy:
            safejsondump(self.statistics, os.path.join(self.result_dir, 'nodule_detection_stat.json'))

if __name__ == '__main__':
    sys.exit(NoduleDetectApp().main() or 0)

# eof
