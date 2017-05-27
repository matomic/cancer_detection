# -*- coding: utf-8 -*-
'''Handles basic configuration from console/CLI'''
from __future__ import print_function
from __future__ import division

import argparse
import collections
import os

import pandas as pd

# namedtuple types to be shared by all configures
DATA_ROOT = os.environ.setdefault('LUNA_DIR', '/home/qiliu/Share/Clinical/lung/luna16/')

NetSpecs = collections.namedtuple('_NetSpecType', ('name', 'version', 'params'))
DirsList = collections.namedtuple('_DTypes', ('data_dir', 'cache_dir', 'params_dir', 'res_dir', 'log_dir', 'csv_dir'))


class PipelineApp(object):
    '''                         (つ・▽・)つ⊂(・▽・⊂)
    Console app wrapper, subclass generally represent a step in the training pipeline.
    Base class contains boilerplate methods for handling command line argument
    parsing, loading appropriate config*.py, managing/provisioning directories
    etc.

    - Command line argument parsing.
      Subclass can extends list of CLI flags, by overwriting the arg_parser
      method, obtain parent parsing by calling _get_parent_arg_parser, then add
      additional arguments to the parser and return it.
    - CLI entry point
      Subclass should implements `_main_impl` method.  `.main()` contains
      boilerplate code for parsing commandling arguments, saving result to
      self.parsedArgs, then calling instance `_main_impl`.
    '''
    def __init__(self):
        self.parsedArgs = None
        self.unet_cfg   = None # configuration data for unet
        self.n3d_cfg    = None # configuration data for 3D nodule net

        # Generate each pipeline step involves using files from some input
        # directory and generating output in some other directory.
        self.input_dir = None
        self._reslt_dir = None

    root = os.environ.setdefault('LUNA_DIR', '/home/qiliu/Share/Clinical/lung/luna16/')
    dirs = DirsList(
            data_dir   = os.path.join(root, 'data'),
            cache_dir  = os.path.join(root, 'cache'),
            params_dir = os.path.join(root, 'params'),
            res_dir    = os.path.join(root, 'results'),
            log_dir    = os.path.join(root, 'log'),
            csv_dir    = os.path.join(root, 'CSVFILES'),
            )

    @property
    def result_dir(self):
        '''auto-privisioned result directory'''
        self.provision_dirs(self._reslt_dir)
        return self._reslt_dir

    checkpoint_path_fmrstr = '{net}_fold{fold}.hdf5'
    def checkpoint_path(self, net, fold, **kwds):
        '''Return the check-point file for CV {fold} for network named {net}

        If --chkpt_path points to a path that is not a directory, use it as is
        If --chkpt-path does not contain /, returns in result-dir/chkpt-path
        If --chkpt-path is unspecified, use default construction based on config
        '''
        assert net, 'ill-defined net name'
        return os.path.join(self.result_dir,
                self.checkpoint_path_fmrstr.format(net=net, fold=fold, **kwds))

    @staticmethod
    def arg_parser():
        '''Return list of ArgumentParser, of this class, and all the parents'''
        parser = argparse.ArgumentParser()

        parser.add_argument('--config-unet', action='store', default=None,
                help='explicit config.py file to be imported, otherwise depends on tags.')

        parser.add_argument('--config-n3d', action='store', default=None,
                help='explicit config.py file to be imported for 3D net, otherwise depends on tags.')

        parser.add_argument('--subset', action='append', default=None, type=int,
                help='when specified, limited script to data from single subset')

        parser.add_argument('--hdf5', action='store_true', default=None,
                help='Use HDF5 pipeline (preprocessed data are stored in HDF5 file, not complete).')

        group = parser.add_mutually_exclusive_group()
        group.add_argument('--session', action='store', default=None,
                help='session name')

        group.add_argument('--result-dir', action='store',
                help='Overwrite default dirs.res_dir where results are stored')

        #parser.add_argument('tags', type=str, action='append',
        #        help='config tags')

        #parser.add_argument('--pretend', '-n', action='store_true',
        #        help='Do not write result to disk')

        return parser

    def argparse_parse_args(self, sys_argv=None):
        '''Parse command line arguments'''
        parser = self.arg_parser()
        parsedArgs = parser.parse_args(sys_argv)

        self.parsedArgs = parsedArgs

    def argparse_postparse(self, parsedArgs=None):
        '''Actions to perform after CLI arguments are parsed'''
        if parsedArgs is None:
            parsedArgs = self.parsedArgs
        # Load value for self.unet_cfg
        #self._parsedArgs_config(parsedArgs)
        if parsedArgs.config_unet is not None:
            if not parsedArgs.config_unet.startswith('config_'):
                parsedArgs.config_unet = 'config_' + parsedArgs.config_unet # support both --config-unet config_v5 and --config-unet v5
            self.unet_cfg = __import__(parsedArgs.config_unet)
        if parsedArgs.config_n3d is not None:
            if not parsedArgs.config_n3d.startswith('config_'):
                parsedArgs.config_n3d = 'config_' + parsedArgs.config_n3d
            self.n3d_cfg  = __import__(parsedArgs.config_n3d)
        self._reslt_dir = parsedArgs.result_dir or self.dirs.res_dir
        if parsedArgs.session:
            self._reslt_dir = os.path.join(self._reslt_dir, parsedArgs.session)

    @staticmethod
    def provision_dirs(*dirs):
        '''Create all the needed dirs'''
        for d in dirs:
            if not os.path.isdir(d):
                os.makedirs(d)

    @staticmethod
    def subst_ext(path, ext):
        '''substitute file {ext}ention of {path}'''
        return os.path.splitext(path)[0] + '.' + ext.lstrip('.')

    def main(self, sys_argv=None):
        '''boilerplate main'''
        self.argparse_parse_args(sys_argv)
        self.argparse_postparse(self.parsedArgs)
        self._main_impl()

    def _main_impl(self):
        '''actual main implementations'''
        raise NotImplementedError

# eof
