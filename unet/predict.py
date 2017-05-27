from __future__ import division, print_function

import os.path
import sys
import itertools
import glob

#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt
import numpy as np
import simplejson as json

import SimpleITK as sitk

from keras.backend import tensorflow_backend as K

# in-house
from NoduleDetect import NoduleDetectApp
from console import PipelineApp
from train import unet_from_checkpoint
from trainNodule import n3dnet_from_checkpoint

from pprint  import pprint
from ipdb import set_trace

npf32_t = np.float32 # pylint: disable=no-member

def glob_multiple(*patterns):
    '''chain {patterns} glob's'''
    return itertools.chain.from_iterable(glob.glob(p) for p in patterns)

class Case(object):
    '''Represent a patient case'''
    @classmethod
    def fromLuNA16(cls, mhd_path):
        '''/.../subset{subset}/{suid}.mhd with associated .raw file'''
        assert os.path.isfile(mhd_path), 'not a file: {}'.format(mhd_path)
        head, suid_mhd = os.path.split(mhd_path)
        subset = os.path.basename(head)
        subset = int(subset[6:])
        suid = os.path.splitext(suid_mhd)[0]
        return cls(suid, mhd_path, subset=subset)

    def __init__(self, suid, img_file, **kwds):
        self.suid = suid
        self.img_file = img_file
        self.extra_attrs = kwds

        # populate by calling self.readImageFile
        self._image  = None
        self.origin  = None
        self.spacing = None
        self.h = None

    @property
    def image(self):
        '''read and return image array data'''
        if self._image is None:
            self.readImageFile()
        return self._image

    def readImageFile(self):
        '''read in image file and populate relevant attributes'''
        # load iamges
        itkimage = sitk.ReadImage(self.img_file)
        self._image  = sitk.GetArrayFromImage(itkimage) # axis, sagittal, coronal
        self.origin  = np.array(itkimage.GetOrigin()) #x,y,z
        self.spacing = np.array(itkimage.GetSpacing())
        self.h = self.spacing[2] / self.spacing[0] # ratio between slice spacing and pixel size


class CancerDetection(PipelineApp):
    '''
    python predict.py --session {session}
    '''
    def arg_parser(self):
        parser = super(CancerDetection, self).arg_parser()

        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--input',
                help="Input file (MHD)")
        group.add_argument('--input-dir',
                help="Input directory (DICOM)")

        return parser

    def _main_impl(self):
        case = Case.fromLuNA16(self.parsedArgs.input)
        self.do_prediction(case, session_dir=self.result_dir, unet_cfg=self.unet_cfg, n3d_cfg=self.n3d_cfg)

    @classmethod
    def get_trained_model(cls, session_dir, net, fold=None, width=None, tag=None):
        '''Load trained model for {net} from {session_dir}'''
        patterns = [os.path.join(session_dir, p) for p in ('{}_*.json'.format(net), '*.json')]
        for checkpoint_json in glob_multiple(*patterns):
            checkpoint_json = json.load(open(checkpoint_json,'r'))
            if (checkpoint_json['__id__']['name'] != net                                       # model name does not match
                    or (fold  is not None and checkpoint_json['__id__']['fold']  != int(fold)) # fold does not match
                    or (tag   is not None and checkpoint_json['__id__']['tag']   != tag)       # tag does not match
                    or (width is not None and checkpoint_json['config']['WIDTH'] != width)):   # input WIDTH does not match
                continue
            checkpoint_path = checkpoint_json['model']['checkpoint_path']
            if not os.path.isabs(checkpoint_path):
                checkpoint_path = os.path.join(session_dir, checkpoint_path)
            if not os.path.isfile(checkpoint_path):
                print('checkpoint path {} is not a valid file: training failed?'.format(checkpoint_path))
                continue
            config = checkpoint_json['config']
            if net == 'unet':
                model = unet_from_checkpoint(checkpoint_path, config['loss_args'])
            elif net == 'n3dNet':
                model = n3dnet_from_checkpoint(checkpoint_path)
            else:
                raise ValueError("unknown network name {}".format(net))
            return model, config
        raise ValueError('cannot find trained model for network {}, fold={}, width={}, tag={} in {}'.format(net, fold, width, tag, session_dir))

    @classmethod
    def do_prediction(cls, case, session_dir, unet_cfg=None, n3d_cfg=None):
        '''Run nodule prediction on {case} using trained result from {session_dir}'''
        sess = K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        # Run Unet
        with K.tf.device('/gpu:0'):
            K.set_session(sess)
            if unet_cfg is not None:
                fold  = unet_cfg.fitter['folds'][0]
                width = unet_cfg.WIDTH
                tag   = unet_cfg.tag
            else:
                fold = width = tag = None
            unet_model, unet_cfg = cls.get_trained_model(session_dir, 'unet', fold=fold, width=width, tag=tag)
            unet_input, unet_out = NoduleDetectApp.predict_nodules(case, [unet_model])
        candidate_list, _ = NoduleDetectApp.detect_nodule_candidate(unet_input, unet_out, (64, 16))
        # Run nodule 3D net
        with K.tf.device('/gpu:0'):
            K.set_session(sess)
            if n3d_cfg is not None:
                fold  = n3d_cfg.fitter['folds'][0]
                width = n3d_cfg.WIDTH
                tag   = n3d_cfg.tag
            else:
                fold = width = tag = None
            n3d_model, n3d_cfg = cls.get_trained_model(session_dir, 'n3dNet', fold=fold, width=width, tag=tag)
            n3d_in = np.asarray([x.volume_array for x in candidate_list])
            n3d_out = n3d_model.predict(np.expand_dims(n3d_in,-1), batch_size=8)
        for candidate, probability in itertools.izip(candidate_list, n3d_out):
            print("({}, {}, {}) -> {}".format(candidate.x, candidate.y, candidate.z, probability))
        return candidate_list, n3d_out

if __name__ == '__main__':
    sys.exit(CancerDetection().main() or 0)
