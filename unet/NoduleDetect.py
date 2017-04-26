# -*- coding: utf-8 -*-
from __future__ import print_function, division

import functools
import glob
import os
import pickle
import sys
import itertools

from keras.models import load_model
from skimage import measure
#from tqdm import tqdm
#import nibabel as nib
import numpy as np
import pandas as pd

from img_mask_gen import iterLunaCases
from utils import dice_coef

npf32_t = np.float32

### loss fuction to make getmodel work, copied from segment/utils.py
def getmodel(tag, fold, cfg):
    '''Return the 2D Unet model of {tag} (version) with {fold}'''
    loss_func = functools.partial(dice_coef, smooth=10, pred_mul=1.0, p_ave=0.6, negate=True)
    model_path=os.path.join(cfg.params_dir, 'unet_{}_{}_fold{}.hdf5'.format(512, tag, fold))
    model = load_model(model_path, custom_objects={'loss': loss_func})
    return model

def luna_nodule_detect(nodule_case_gen, models, output_shape, output_path, cfg):
    '''ʕ •̀ o •́ ʔ'''
    W, H = output_shape
    for n, case in enumerate(nodule_case_gen):
        h = case.spacing[2] / case.spacing[0]
        res = np.interp(case.image, [-1000, 200], [0, 255], left=0, right=255).astype(np.uint8) # HU -_> [0, 255]
        res = np.array(res*case.lung_mask, npf32_t)*(1.0/255)-0.05
        res = np.expand_dims(res, axis=-1)
        cut = cfg.keep_prob
        nodules = models[0].predict(res,batch_size=8)>cut
        for model in models[1:]:
            nodules += model.predict(res,batch_size=8)>cut
        shape = nodules.shape[:-1]
        nodules = np.reshape(nodules,shape)*(case.lung_mask>0.2)

        ## find nodules and its central location
        binary_nodule = np.array(nodules >=0.5, dtype=np.int8)
        labels = measure.label(binary_nodule)
        vals, counts = np.unique(labels, return_counts=True)
        counts = counts[vals!=0]
        vals = vals[vals!=0]
        nod_list = sorted(itertools.izip(counts,vals), reverse=True)

        ## larger than 100 pixels in volume
        volume_cut = 50
        nod_list = [x for x in nod_list if x[0]>volume_cut]

        nod_res =[]
        ls = []
        Ndeteced = 0
        for nod in nod_list:
            xyz = np.where(labels==nod[1])
            cz, cy, cx = tuple(int(np.median(x)) for x in xyz)
            #ss = tuple(int(np.max(x)-np.min(x))+1 for x in xyz)
            #if ss[0]<3: #thickness<3 pixels
            #    continue
            # output result image with WIDTH
            cy_min, cy_max = cy-W//2, cy+W//2
            cx_min, cx_max = cx-W//2, cx+W//2
            cz_min, cz_max = cz-H//2, cz+H//2

            if cy_min<=0 or cy_max>512 or cx_min<=0 or cx_max>512 or cz_min<=0 or cz_max>=res.shape[0]:
                continue
            try:
                out = np.transpose(res[cz_min:cz_max, cy_min:cy_min, cx_min:cx_max, 0]+0.05, (1,2,0))
            except Exception:
                continue
            nod_res.append(out)
            l = 0
            for xyzd in case.nodules:
                dz = np.abs(xyzd[2]-cz)*h
                dx = np.abs(xyzd[0]-cx)
                dy = np.abs(xyzd[1]-cy)
                if dx*dx+dy*dy+dz*dz <= xyzd[3]**2/4:
                    l=1
                    Ndeteced += 1
                    break
            ls.append(l)
        print('case {:03d}: # nodules {:03d}; # ? {:03d}; detected {:03d}.'.format(n, len(case.nodules), len(ls), Ndeteced))

        # save
        f = os.path.join(output_path,'{}.pkl'.format(case.hashid))
        with open(f, 'wb') as output:
            pickle.dump([nod_res, ls], output)

    return n # number of cases

def nodule_detect_on_luna(models, subset, subset_path, df_node, output_path, output_shape, cfg):
    '''(=^･ｪ･^=)'''
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    cases = luna_nodule_detect(iterLunaCases(subset, subset_path, df_node), models, output_shape, output_path, cfg)

    return cases

def main(sys_argv):
    ## output nodule image size
    W = 64
    H = 16

    assert len(sys.argv)==2
    tags = sys.argv[1].split(',')
    cfg = __import__('config_v{}'.format(tags[0])) # NOTE: need to handle multiple tag case?
    models = [getmodel(int(t), 0, cfg) for t in tags]

    output_path = os.path.join(cfg.root,'nodule_candidate_%s'%('_'.join(tags)))
    df_node = pd.read_csv(os.path.join(cfg.csv_dir, "annotations.csv"))
    for subset in range(10):
        print("processing subset ",subset)
        subset_path = os.path.join(cfg.root,"data","subset{}".format(subset))
        nodule_detect_on_luna(models, subset, subset_path, df_node, output_path, output_shape=(W, H), cfg=cfg)

if __name__ == '__main__':
    exit(main(sys.argv))
