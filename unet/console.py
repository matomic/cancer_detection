# -*- coding: utf-8 -*-
'''Handles basic configuration from console/CLI'''
from __future__ import print_function
from __future__ import division

import argparse
import collections
import datetime
import os
import time

import simplejson as json

from config import UnetConfig, N3DConfig
from utils import safejsondump

# namedtuple types to be shared by all configures
DATA_ROOT = os.environ.setdefault('LUNA_DIR', '/home/qiliu/Share/Clinical/lung/luna16/')

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

		self.session_json = None

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

	@property
	def session_json_path(self):
		'''default path to dump session JSON'''
		return os.path.join(self.result_dir, 'session.json')

	def dump_session(self, session_json=None, path=None):
		'''Dump session json to {path}'''
		if path is None:
			path = self.session_json_path
		if session_json is None:
			session_json = self.session_json
		session_json = session_json.copy()
		now = time.time()
		session_json['__id__'] = {
			    'name' : self.parsedArgs.session,
			    #'tags' : [self.unet_cfg.tag, self.n3d_cfg.tag],
			    'ts'   : now,
			    'dt'   : datetime.datetime.fromtimestamp(now).isoformat()
			}
		safejsondump(session_json, path)

	def load_session(self, path=None):
		'''Load JSON file {path} into self.session_json'''
		if path is None:
			path = self.session_json_path
		if self.session_json is None:
			self.session_json = {}
		if os.path.isfile(path):
			self.session_json.update(json.load(open(path, 'r')))
		return self.session_json

	checkpoint_path_fmrstr = '{net}_fold{fold}.hdf5'
	def checkpoint_path(self, net, fold, **kwds):
		'''
		Return the check-point file for CV {fold} for network named {net}

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

		parser.add_argument('--result-dir', action='store',
		        help='Overwrite default dirs.res_dir where results are stored')

		parser.add_argument('session', action='store', default=None,
		        help='session name')

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
		# Set result directory
		self._reslt_dir = parsedArgs.result_dir or self.dirs.res_dir
		if parsedArgs.session:
			self._reslt_dir = os.path.join(self._reslt_dir, parsedArgs.session)

		# Load network configurations
		self.session_json = self.load_session() or {}
		self.unet_cfg = self._loadNetConfiguration('unet', UnetConfig, source=self.parsedArgs.config_unet)
		self.n3d_cfg  = self._loadNetConfiguration('n3d',  N3DConfig,  source=self.parsedArgs.config_n3d)

	def _loadNetConfiguration(self, net_name, net_cls, source=None):
		net_cfg = self.session_json.setdefault(net_name, {}).get('config')
		if source is not None and (net_cfg is None
				or input("Session {} already has configuration for {}.  Overwrite with config from {}? (y/N)".format(self.parsedArgs.session, net_name, source)).lower().startswith('y')):
			net_cfg = net_cls.fromConfigPy(source)
			self.session_json[net_name]['config'] = net_cfg.toDict()
			print("{} configuration loaded from {}".format(net_name, source))
		elif net_cfg:
			net_cfg = net_cls.fromDict(net_cfg)
			print("{} configuration loaded from session {}".format(net_name, self.parsedArgs.session))
		return net_cfg

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

# eof vim: set noet ci pi sts=0 sw=4 ts=4:
