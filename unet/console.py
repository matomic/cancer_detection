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
        self.cfg = None
        # Generate each pipeline step involves using files from some input
        # directory and generating output in some other directory.
        self.input_dir = None
        self._reslt_dir = None

    @property
    def cfg_json(self):
        '''JSONify self.cfg'''
        return self.cfg.to_json()

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

        parser.add_argument('--config', action='store', default=None,
                help='explicit config.py file to be imported, otherwise depends on tags. ')

        parser.add_argument('--subset', action='append', default=None, type=int,
                help='when specified, limited script to data from single subset')

        parser.add_argument('--hdf5', action='store_true', default=None,
                help='Use HDF5 pipeline (preprocessed data are stored in HDF5 file, not complete).')

        parser.add_argument('--session', action='store', default=None,
                help='session name')

        parser.add_argument('--result-dir', action='store',
                help='Overwrite default config.dirs.res_dir where results are stored')

        parser.add_argument('tags', type=str, action='append',
                help='config tags')

        #parser.add_argument('--pretend', '-n', action='store_true',
        #        help='Do not write result to disk')

        return parser

    def argparse_parse_args(self, sys_argv=None):
        '''Parse command line arguments'''
        parser = self.arg_parser()
        parsedArgs = parser.parse_args(sys_argv)

        self.parsedArgs = parsedArgs

    def _parsedArgs_config(self, parsedArgs=None):
        '''Figure out which config file to import'''
        if parsedArgs is None:
            parsedArgs = self.parsedArgs

        # Figure out parsedArgs.config
        if parsedArgs.config is None:
            parsedArgs.config = 'config_v{}'.format(parsedArgs.tags[0])
        if not parsedArgs.config.startswith('config_'):
            parsedArgs.config = 'config_' + parsedArgs.config
        parsedArgs.config = os.path.splitext(parsedArgs.config)[0]

    def argparse_postparse(self, parsedArgs=None):
        '''Actions to perform after CLI arguments are parsed'''
        if parsedArgs is None:
            parsedArgs = self.parsedArgs
        self._parsedArgs_config(parsedArgs)
        self.cfg = __import__(parsedArgs.config)
        self.cfg.dirs = DirsList(**self.cfg.dirs)
        self._reslt_dir = parsedArgs.result_dir or self.cfg.dirs.res_dir
        if parsedArgs.session:
            self._reslt_dir = os.path.join(self._reslt_dir, parsedArgs.session)

    @staticmethod
    def provision_dirs(*dirs):
        '''Create all the needed dirs'''
        for d in dirs:
            if not os.path.isdir(d):
                os.makedirs(d)

    def main(self, sys_argv=None):
        '''boilerplate main'''
        self.argparse_parse_args(sys_argv)
        self.argparse_postparse(self.parsedArgs)
        self._main_impl()

    def _main_impl(self):
        '''actual main implementations'''
        raise NotImplementedError

# eof
