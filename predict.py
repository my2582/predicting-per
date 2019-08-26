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
import sys
import copy
import math

import pandas as pd
import numpy as np
import tensorflow as tf


from batch_generator import BatchGenerator

import model_utils

def print_vector(name,v):
    print("%s: "%name,end='')
    for i in range(len(v)):
        print("%.2f "%v[i],end=' ')
    print()

def predict(config):

    target_list = ['saleq_ttm', 'cogsq_ttm',
       'xsgaq_ttm', 'oiadpq_ttm', 'mkvaltq_ttm', 'niq_ttm', 'ibq_ttm',
       'cheq_mrq', 'rectq_mrq', 'invtq_mrq', 'acoq_mrq', 'ppentq_mrq',
       'aoq_mrq', 'dlcq_mrq', 'apq_mrq', 'txpq_mrq', 'lcoq_mrq', 'ltq_mrq']
    aux_list = ['mom1m','mom3m', 'mom6m', 'mom9m']
    df = pd.DataFrame(columns=['date', 'gvkey', 'mse', 'normalizer',
       config.target_field+"_output", config.target_field+"_target"])
    datafile = config.datafile
    
    if config.predict_datafile is not None:
        datafile = config.predict_datafile

    print("Loading data from %s ..."%datafile)
    path = os.path.join(config.data_dir,datafile)
    
    ind = 0
    
    config.batch_size = 1
    batches = BatchGenerator(path, 
                             config,
                             require_targets=config.require_targets, 
                             verbose=True)
    batches.cache(verbose=True)

    tf_config = tf.ConfigProto( allow_soft_placement=True  ,
                                log_device_placement=False )

    index = int(np.argwhere(np.array(target_list) == config.target_field).mean())
    with tf.Graph().as_default(), tf.Session(config=tf_config) as session:

        model = model_utils.get_model(session, config, verbose=True)

        perfs = dict()

        for i in range(batches.num_batches):
            batch = batches.next_batch()

            (mse, preds) = model.step(session, batch)
            # (mse, preds) = model.debug_step(session, batch)

            if math.isnan(mse) is False:
                date = batch_to_date(batch)
                if date not in perfs:
                    perfs[date] = list()
                perfs[date].append(mse)

            if config.pretty_print_preds is True:
                #pretty_print_predictions(batches, batch, preds, mse)
                key     = batch_to_key(batch)
                date    = batch_to_date(batch)
                if int(date%100) in [3,6,9,12]:
                    print("GVKEY: " + str(key) + ", Date: " + str(date))
                    L = batch.seq_lengths[0]
                    targets = batch.targets[L-1][0]
                    outputs = preds[0]
                    normalizer = batch.normalizers[0]
                
                    np.set_printoptions(suppress=True)
                    np.set_printoptions(precision=3)
                    
                    df.loc[ind] = [date, key, mse, normalizer, 
                          batches.get_raw_outputs(batch,0,outputs)[index],
                          batches.get_raw_outputs(batch,0,targets)[index]]
                    ind += 1

            else:
                print_predictions(batches, batch, preds)

        if config.mse_outfile is not None:
            with open(config.mse_outfile,"w") as f:
                for date in sorted(perfs):
                    mean = np.mean(perfs[date])
                    print("%s %.6f %d"%(date,mean,len(perfs[date])),file=f)
                total_mean = np.mean( [x for v in perfs.values() for x in v] )
                print("Total %.6f"%(total_mean),file=f)
            df.to_csv('datasets/' + config.output_file, index=False)
            f.closed
        else:
            df.to_csv('datasets/' + config.output_file, index=False)
            exit()


def batch_to_key(batch):
    idx = batch.seq_lengths[0]-1
    assert( 0<= idx )
    assert( idx < len(batch.attribs) )
    return batch.attribs[idx][0][0]

def batch_to_date(batch):
    idx = batch.seq_lengths[0]-1
    assert( 0<= idx )
    assert( idx < len(batch.attribs) )
    if (batch.attribs[idx][0] is None):
        print(idx)
        exit()
    return batch.attribs[idx][0][1]

def pretty_print_predictions(batches, batch, preds, mse):
    key     = batch_to_key(batch)
    date    = batch_to_date(batch)

    L = batch.seq_lengths[0]
    targets = batch.targets[L-1][0]
    outputs = preds[0]
    normalizer = batch.normalizers[0]

    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=3)

    print("%s %s mse %.8f %.2f"%(date,key,mse,normalizer))
    inputs = batch.inputs
    for i in range(L):
        print_vector("input[t-%d]"%(L-i-1),batches.get_raw_inputs(batch,0,inputs[i][0]) )
    print_vector("output[t+1]", batches.get_raw_outputs(batch,0,outputs) )
    print_vector("target[t+1]", batches.get_raw_outputs(batch,0,targets) )
    print("--------------------------------")
    sys.stdout.flush()

def print_predictions(batches, batch, preds):
    key     = batch_to_key(batch)
    date    = batch_to_date(batch)
    inputs  = batch.inputs[-1][0]
    outputs = preds[0]

    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=3)
    out = batches.get_raw_outputs(batch,0,outputs)
    out_str = ' '.join(["%.3f"%out[i] for i in range(len(out))])

    print("%s %s %s"%(date,key,out_str))
    sys.stdout.flush()
