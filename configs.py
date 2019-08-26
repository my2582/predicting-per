# -*- coding: utf-8 -*-
"""
Created on Thu May 16 10:48:06 2019

@author: user
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

class _LoadFromFile (argparse.Action):
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            parser.parse_known_args(f.read().split(), namespace)

_global_parser = argparse.ArgumentParser()

_global_parser.add_argument('--config', type=open,
                            action=_LoadFromFile,
                            help="File containing configuration")

class ConfigValues(object):
    def __init__(self):
        self.__dict__['__configs'] = {}
        self.__dict__['__parsed'] = False

    def _parse_configs(self):
        result, _ = _global_parser.parse_known_args()
        if '__configs' not in self.__dict__:
            self.__dict__['__configs'] = {}
        if '__parsed' not in self.__dict__:
            self.__dict__['__parsed'] = False
        for config_name, val in vars(result).items():
            self.__dict__['__configs'][config_name] = val
        self.__dict__['__parsed'] = True

    def __getattr__(self, name):
        if ('__parsed' not in self.__dict__) or (not self.__dict__['__parsed']):
            self._parse_configs()
        if name not in self.__dict__['__configs']:
            raise AttributeError(name)
        return self.__dict__['__configs'][name]

    def __setattr__(self, name, value):
        if ('__parsed' not in self.__dict__) or (not self.__dict__['__parsed']):
            self._parse_configs()
        self.__dict__['__configs'][name] = value



def _define_helper(config_name, default_value, docstring, configtype):
    _global_parser.add_argument("--" + config_name,
                                default=default_value,
                                help=docstring,
                                type=configtype)

def DEFINE_string(config_name, default_value, docstring):
    _define_helper(config_name, default_value, docstring, str)


def DEFINE_integer(config_name, default_value, docstring):
    _define_helper(config_name, default_value, docstring, int)


def DEFINE_boolean(config_name, default_value, docstring):
    def str2bool(v):
        return v.lower() in ('true', 't', '1')
    _global_parser.add_argument('--' + config_name,
                                nargs='?',
                                const=True,
                                help=docstring,
                                default=default_value,
                                type=str2bool)
    _global_parser.add_argument('--no' + config_name,
                                action='store_false',
                                dest=config_name)
DEFINE_bool = DEFINE_boolean 

def DEFINE_float(config_name, default_value, docstring):
    _define_helper(config_name, default_value, docstring, float)
