# -*- coding: utf-8 -*-
"""
Created on Thu May 16 10:48:06 2019

@author: user
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os

import numpy as np
import tensorflow as tf
import pickle

from tensorflow.python.platform import gfile
from deep_rnn_model import DeepRnnModel

def save_model(session, config, step):
    last_checkpoint_path = tf.train.latest_checkpoint(config.model_dir)
    checkpoint_path = os.path.join(config.model_dir, "training.ckpt")
    tf.train.Saver().save(session, checkpoint_path, global_step=step)
    if last_checkpoint_path is not None:
        os.remove(last_checkpoint_path+'.data-00000-of-00001')
        os.remove(last_checkpoint_path+'.index')
        os.remove(last_checkpoint_path+'.meta')

def adjust_learning_rate(session, model,
                         learning_rate, lr_decay, cost_history, lookback=5):

    lookback += 1
    if len(cost_history) >= lookback:
        mean = np.mean(cost_history[-lookback:-2])
        curr = cost_history[-1]
        # If performance has dropped by less than 1%, decay learning_rate
        if ((learning_rate >= 0.0001) and (mean > 0.0)
            and (mean >= curr) and (curr/mean >= 0.98)):
            learning_rate = learning_rate * lr_decay
    model.set_learning_rate(session, learning_rate)
    return learning_rate

def get_scaling_params(config, data, verbose=False):
    scaling_params = None
    if config.scalesfile is not None and os.path.isfile(config.scalesfile):
        scaling_params = pickle.load( open( config.scalesfile, "rb" ) )
        if verbose:
            print("Reading scaling params from %s"%config.scalesfile);
    else:
        scaling_params = data.get_scaling_params(config.data_scaler)
        if config.scalesfile is not None:
            pickle.dump(scaling_params, open( config.scalesfile, "wb" ))
            if verbose:
                print("Writing scaling params to %s"%config.scalesfile);

    if verbose:
        print("Scaling params are:")
        print("%-10s %-6s %-6s"%('feature','mean','std'))
        for i in range(len(data.feature_names)):
            center = "%.4f"%scaling_params['center'][i];
            scale  = "%.4f"%scaling_params['scale'][i];
            print("%-10s %-6s %-6s"%(data.feature_names[i],
                                     center,scale))
    return scaling_params


def get_model(session, config, verbose=False):
    model = _create_model(session, config, verbose)

    ckpt = tf.train.get_checkpoint_state(config.model_dir)
    start_time = time.time()
    if ckpt and gfile.Exists(ckpt.model_checkpoint_path+".index"):
        if verbose:
            print("Reading model parameters from {}...".format(
                ckpt.model_checkpoint_path), end=' ')
        tf.train.Saver(max_to_keep=200).restore(session,
                                                ckpt.model_checkpoint_path)
        if verbose:
            print("done in %.2f seconds."%(time.time() - start_time))
    else:
        if verbose:
            print("Creating model with fresh parameters ...", end=' ')
        session.run(tf.global_variables_initializer())
        if verbose:
            print("done in %.2f seconds."%(time.time() - start_time))

    return model

def _create_model(session,config,verbose=False):

    if verbose is True:
        print("Model has the following geometry:")
        print("  model_type  = %s"% config.nn_type)
        print("  min_unroll  = %d"% config.min_unrollings)
        print("  max_unroll  = %d"% config.max_unrollings)
        print("  stride      = %d"% config.stride)
        print("  batch_size  = %d"% config.batch_size)
        print("  num_inputs  = %d"% config.num_inputs)
        print("  num_outputs = %d"% config.num_outputs)
        print("  num_hidden  = %d"% config.num_hidden)
        print("  num_layers  = %d"% config.num_layers)
        print("  optimizer   = %s"% config.optimizer)
        print("  device      = %s"% config.default_gpu)

    initer = tf.random_uniform_initializer(-config.init_scale,config.init_scale,seed=config.seed)

    with tf.variable_scope("model", reuse=None, initializer=initer), tf.device(config.default_gpu):

        model = DeepRnnModel(config)

    return model
