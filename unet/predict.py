from __future__ import division, print_function

import os.path
import sys
import itertools
import glob
import tempfile
import zipfile

#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt
import dicom
import numpy as np
import simplejson as json

import SimpleITK as sitk

from keras.backend import tensorflow_backend as K

# in-house
from .NoduleDetect import NoduleDetectApp
from .console import PipelineApp
from .train import unet_from_checkpoint
from .trainNodule import n3dnet_from_checkpoint

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

    @classmethod
    def fromZip(cls, zip_path):
        suid = os.path.splitext(os.path.basename(zip_path))[0]
        with tempfile.TemporaryDirectory() as tmp_dir:
            zf = zipfile.ZipFile(zip_path, 'r')
            zf.extractall(tmp_dir)
            mhd_file = os.path.join(tmp_dir, '{}.mhd'.format(suid))
            if os.path.isfile(mhd_file):
                return cls(suid, mhd_file)
            for mhd_file in glob.glob(os.path.join(tmp_dir, '*.mhd')):
                return cls(suid, mhd_file)
            raise IOError("cannot find .mhd file in archive")

    def __init__(self, suid, img_file, **kwds):
        self.suid = suid
        self.img_file = img_file
        self.extra_attrs = kwds

        # load iamges
        self._itkimage = itkimage = sitk.ReadImage(self.img_file)
        self.image   = sitk.GetArrayFromImage(itkimage) # axis, sagittal, coronal
        self.origin  = np.array(itkimage.GetOrigin()) #x,y,z
        self.spacing = np.array(itkimage.GetSpacing())
        self.h = self.spacing[2] / self.spacing[0] # ratio between slice spacing and pixel size


class DicomCase(object):
    '''Represent a patient case from DICOM'''
    @classmethod
    def fromPathIter(cls, file_iter):
        dcm_list = []
        suid_set = set()
        for f in file_iter:
            try:
                dcm = dicom.read_file(f)
            except:
                continue
            if dcm.Modality != 'CT':
                continue
            dcm_list.append(dcm)
            suid_set.add(dcm.SeriesInstanceUID)
        assert len(suid_set) == 1, 'none unique SeriesInstanceUID found: {}'.format(suid_set)
        return cls(suid_set.pop(), dcm_list)

    @classmethod
    def fromDir(cls, dcm_dir):
        '''/////'''
        return cls.fromPathIter(glob.glob(os.path.join(dcm_dir, '*')))

    @classmethod
    def fromZip(cls, zip_path):
        '''\\\\'''
        with tempfile.TemporaryDirectory() as tmp_dir:
            zf = zipfile.ZipFile(zip_path, 'r')
            zf.extractall(tmp_dir)
            dcm_file_list = [os.path.join(root,f) for root, _, file_list in os.walk(tmp_dir) for f in file_list]
            return cls.fromPathIter(dcm_file_list)

    def __init__(self, suid, dcm_list, **kwds):
        '''
        :param dcm_list: list of DICOM data sorted by Slice Location or ImagePositionPatient
        '''
        self.suid = suid
        self.extra_attrs = kwds

        ct_hu_npy_list = []
        imgpost_npy = []
        spacing_npy = []
        pxsize_set = set()
        dcm_list = sorted(dcm_list, key=lambda x : x.InstanceNumber)
        for n, dcm in enumerate(dcm_list):
            ct_hu_npy_list.append(dcm.RescaleSlope*dcm.pixel_array + dcm.RescaleIntercept)
            pxsize_set.add((*dcm.PixelSpacing,))
            spacing_npy.append(dcm.ImagePositionPatient[-1])
            imgpost_npy.append(dcm.ImagePositionPatient)

        self.image = np.asarray(ct_hu_npy_list, dtype='int16')
        self.origin = [float(x) for x in imgpost_npy[0]]

        spacing_npy = np.abs(np.diff(spacing_npy))
        assert len(pxsize_set) == 1 and spacing_npy.max() - spacing_npy.min() < 1e-2, 'uneven spacing is not supported.'
        self.spacing = np.asarray([*pxsize_set.pop(), np.median(spacing_npy)])
        self.h = self.spacing[2] / self.spacing[0]


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
    def candidate_stage(cls, case, sess, session_dir, unet_cfg=None):
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
            try:
                unet_input, unet_out = NoduleDetectApp.predict_nodules(case, [unet_model])
            finally:
                del unet_model
        candidate_list, _ = NoduleDetectApp.detect_nodule_candidate(unet_input, unet_out, (64, 16))
        return candidate_list, unet_cfg

    @classmethod
    def probability_stage(cls, candidate_list, sess, session_dir, n3d_cfg=None):
        with K.tf.device('/gpu:0'):
            K.set_session(sess)
            if n3d_cfg is not None:
                fold  = n3d_cfg.fitter['folds'][0]
                width = n3d_cfg.WIDTH
                tag   = n3d_cfg.tag
            else:
                fold = width = tag = None
            n3d_model, n3d_cfg = cls.get_trained_model(session_dir, 'n3dNet', fold=fold, width=width, tag=tag)
            try:
                n3d_in = np.asarray([x.volume_array for x in candidate_list])
                n3d_out = n3d_model.predict(np.expand_dims(n3d_in,-1), batch_size=8)
            finally:
                del n3d_model
        for candidate, probability in zip(candidate_list, n3d_out):
            print("({}, {}, {}) -> {}".format(candidate.x, candidate.y, candidate.z, probability))
        return n3d_out, n3d_cfg

    @classmethod
    def do_prediction(cls, case, session_dir, unet_cfg=None, n3d_cfg=None, device='gpu'):
        '''Run nodule prediction on {case} using trained result from {session_dir}'''
        K.tf.reset_default_graph()
        sess = K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        try:
            # Run Unet to generate list of candidates
            candidate_list, unet_cfg = cls.candidate_stage(case, sess, session_dir, unet_cfg)
            if not candidate_list:
                print("did not find any nodule candidate.")
                return [], []
            # Run N3D net to generate candidate is nodule probability
            n3d_out, n3d_cfg = cls.probability_stage(candidate_list, sess, session_dir, n3d_cfg)
        finally:
            sess.close()
        return candidate_list, n3d_out

if __name__ == '__main__':
    sys.exit(CancerDetection().main() or 0)
