'''Handles basic configuration from console/CLI'''
from __future__ import print_function
from __future__ import division

import argparse
import collections
import os

# namedtuple types to be shared by all configures
NetSpecs = collections.namedtuple('_NetSpecType', ('name', 'version', 'params'))
DirsList = collections.namedtuple('_DTypes', ('cache_dir', 'params_dir', 'res_dir', 'log_dir', 'csv_dir'))

def parse_args(sys_argv=None, description=''):
    '''Parse command line arguments'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', action='store', default=None,
            help='config.py file to be imported')
    parser.add_argument('tags',
            help='config tags',
            nargs='+')
    parsedArgs = parser.parse_args(sys_argv)
    if parsedArgs.config is None:
        parsedArgs.config = 'config_{}'.format(parsedArgs.tags[0])
    parsedArgs.config = os.path.basename(os.path.splitext(parsedArgs.config)[0])

    return parsedArgs

# eof
