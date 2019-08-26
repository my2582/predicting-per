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

import tensorflow as tf
import regex as re
import math

import model_utils
from batch_generator import BatchGenerator
import configs as configs

#%%
def get_configs():
    configs.DEFINE_string("name",'trial1',"")
    configs.DEFINE_string("datafile", 'Group2-Dataset.csv', "")
    configs.DEFINE_string("predict_datafile", None, "")
    configs.DEFINE_string("mse_outfile", None, "")
    configs.DEFINE_string("scalesfile", None, "")
    configs.DEFINE_string("default_gpu", '/gpu:0', "")
    configs.DEFINE_string("nn_type",'DeepRnnModel',"")
    configs.DEFINE_string("active_field", 'active', "")
    configs.DEFINE_string("date_field", 'date', "")
    configs.DEFINE_string("key_field", 'gvkey',"")
    configs.DEFINE_string("target_field", 'mkvaltq_ttm',"")
    configs.DEFINE_string("scale_field", 'mrkcap',"")
    configs.DEFINE_string("financial_fields", 'saleq_ttm-ltq_mrq',"")
    configs.DEFINE_string("aux_fields", 'mom3m-mom9m', "")
    configs.DEFINE_string("dont_scale", None,"")
    configs.DEFINE_string("data_dir",'datasets',"")
    configs.DEFINE_string("model_dir",'chkpts-wrds-rnn',"")
    configs.DEFINE_string("rnn_cell",'lstm',"")
    configs.DEFINE_string("activation_fn",'relu',"")
    configs.DEFINE_integer("num_inputs", -1,"")
    configs.DEFINE_integer("num_outputs", -1,"")
    configs.DEFINE_integer("target_idx",None,"")
    configs.DEFINE_integer("min_unrollings",5,"")
    configs.DEFINE_integer("max_unrollings",5,"")
    configs.DEFINE_integer("min_years",None,"")
    configs.DEFINE_integer("max_years",None,"")
    configs.DEFINE_integer("pls_years",None,"")

    configs.DEFINE_integer("num_unrollings",5,"")
    configs.DEFINE_integer("stride",12,"")
    configs.DEFINE_integer("forecast_n",3,"")
    configs.DEFINE_integer("batch_size",128,"")
    configs.DEFINE_integer("num_layers",5, "")
    configs.DEFINE_integer("num_hidden",128,"")
    configs.DEFINE_float("training_noise",None, "")
    configs.DEFINE_float("init_scale",0.01, "")
    configs.DEFINE_float("max_grad_norm",10.0,"")
    configs.DEFINE_integer("start_date",None,"")
    configs.DEFINE_integer("end_date",None,"")
    configs.DEFINE_integer("split_date",None,"")
    configs.DEFINE_float("keep_prob",0.75,"")
    configs.DEFINE_boolean("train",False,"")
    configs.DEFINE_boolean("require_targets",False,"")
    configs.DEFINE_boolean("input_dropout",False,"")
    configs.DEFINE_boolean("hidden_dropout",False,"")
    configs.DEFINE_boolean("rnn_dropout",True,"")
    configs.DEFINE_boolean("skip_connections",False,"")
    configs.DEFINE_boolean("direct_connections",False,"")
    configs.DEFINE_boolean("use_cache",True,"")
    configs.DEFINE_boolean("pretty_print_preds",True,"")
    configs.DEFINE_boolean("scale_targets",True,"")
    configs.DEFINE_boolean("backfill",False,"")
    configs.DEFINE_boolean("log_squasher",True,"")
    configs.DEFINE_boolean("ts_smoother",False,"")
    configs.DEFINE_string("data_scaler",'RobustScaler','')
    configs.DEFINE_string("optimizer", 'AdadeltaOptimizer', '')
    configs.DEFINE_string("optimizer_params",None, '')
    configs.DEFINE_float("learning_rate",0.6,"")
    configs.DEFINE_float("lr_decay",0.95, "")
    configs.DEFINE_float("validation_size",0.3,"")
    configs.DEFINE_float("train_until",0.0,"")
    configs.DEFINE_float("passes",0.2,"")
    configs.DEFINE_float("target_lambda",0.8,"")
    configs.DEFINE_float("rnn_lambda",0.2,"")
    configs.DEFINE_float("l2_alpha",0.0,"")
    configs.DEFINE_integer("max_epoch",1000,"")
    configs.DEFINE_integer("early_stop",10,"")
    configs.DEFINE_integer("seed",100,"")
    configs.DEFINE_integer("cache_id",100,"")
    configs.DEFINE_string("output_file", "mkvaltq_2016.csv", "")
    
    c = configs.ConfigValues()

    if c.min_unrollings is None:
        c.min_unrollings = c.num_unrollings

    if c.max_unrollings is None:
        c.max_unrollings = c.num_unrollings

    if c.min_years is not None:
        c.min_unrollings = c.min_years * ( 12 // c.stride )
        if c.max_years is not None:
            c.max_unrollings = (c.max_years) * ( 12 // c.stride )
        elif c.pls_years is None:
            c.max_unrollings = c.min_unrollings
        else:
            c.max_unrollings = (c.min_years+c.pls_years) * ( 12 // c.stride )

    # optimizer_params is a string of the form "param1=value1,param2=value2,..."
    # this maps it to dictionary { param1 : value1, param2 : value2, ...}
    if c.optimizer_params is None:
        c.optimizer_params = dict()
    else:
        args_list = [p.split('=') for p in c.optimizer_params.split(',')]
        params = dict()
        for p in args_list:
            params[p[0]] = float(p[1])
        c.optimizer_params = params
        assert('learning_rate' not in c.optimizer_params)

    return c

#%%
def pretty_progress(step, prog_int, dot_count):
    if ( (prog_int<=1) or (step % (int(prog_int)+1)) == 0):
        dot_count += 1; print('.',end=''); sys.stdout.flush()
    return dot_count

def run_epoch(session, model, train_data, valid_data,
              keep_prob=1.0, passes=1.0, 
              noise_model=None, verbose=False):

    if not train_data.num_batches > 0:
        raise RuntimeError("batch_size*max_unrollings is larger "
                             "than the training set size.")

    start_time = time.time()
    train_mse = valid_mse = 0.0
    dot_count = 0
    train_steps = int(passes*train_data.num_batches)
    valid_steps = valid_data.num_batches
    total_steps = train_steps+valid_steps
    prog_int = total_steps/100 

    train_data.shuffle() 
    valid_data.rewind()

    print("Steps: %d "%total_steps,end=' ')

    for step in range(train_steps):
        batch = train_data.next_batch()
        train_mse += model.train_step(session, batch, keep_prob=keep_prob)
        if verbose: dot_count = pretty_progress(step,prog_int,dot_count)

    # evaluate validation data
    for step in range(valid_steps):
        batch = valid_data.next_batch()
        (mse,_) = model.step(session, batch)
        valid_mse += mse
        if verbose: dot_count = pretty_progress(train_steps+step,prog_int,dot_count)

    if verbose:
        print("."*(100-dot_count),end='')
        print(" passes: %.2f  "
              "speed: %.0f seconds" % (passes,(time.time() - start_time)) )
    sys.stdout.flush()

    return (train_mse/train_steps,valid_mse/valid_steps)

def stop_training(config, perfs):
    """
    Early stop algorithm
    Args:
      config:
      perfs: History of validation performance on each iteration
      file_prefix: how to name the chkpnt file
    """
    window_size = config.early_stop
    if ( (window_size is not None)
     and (len(perfs) > window_size)
     and (min(perfs) < min(perfs[-window_size:])) ):
        return True
    elif config.train_until > perfs[-1]:
        return True
    else:
        return False

def train_model(config):
    if config.start_date is not None:
        print("Training start date: ", config.start_date)
    if config.start_date is not None:
        print("Training end date: ", config.end_date)

    print("Loading training data from %s ..."%config.datafile)
    train_data = None
    valid_data = None

    data_path = os.path.join(config.data_dir, config.datafile)
    batches = BatchGenerator(data_path, config, is_training_only=True)

    train_data = batches.train_batches(verbose=True)
    valid_data = batches.valid_batches(verbose=True)
        
    tf_config = tf.ConfigProto(allow_soft_placement=True,
                               log_device_placement=False)

    with tf.Graph().as_default(), tf.Session(config=tf_config) as session:
        if config.seed is not None:
            tf.set_random_seed(config.seed)

        print("Constructing model ...")
        model = model_utils.get_model(session, config, verbose=True)

        params = model_utils.get_scaling_params(config,train_data,verbose=True)
        model.set_scaling_params(session,**params)

        noise_model = None

        if config.early_stop is not None:
            print("Training will early stop without "
              "improvement after %d epochs."%config.early_stop)
        sys.stdout.flush()

        train_history = list()
        valid_history = list()

        lr = model.set_learning_rate(session,config.learning_rate)

        train_data.cache(verbose=True)
        valid_data.cache(verbose=True)

        for i in range(config.max_epoch):

            (train_mse, valid_mse) = run_epoch(session, model, train_data, valid_data,
                                               keep_prob=config.keep_prob, 
                                               passes=config.passes,
                                               noise_model=noise_model,
                                               verbose=True)
            print( ('Epoch: %d Train MSE: %.6f Valid MSE: %.6f Learning rate: %.4f') %
                  (i + 1, train_mse, valid_mse, lr) )
            sys.stdout.flush()

            train_history.append( train_mse )
            valid_history.append( valid_mse )

            if re.match("Gradient|Momentum",config.optimizer):
                lr = model_utils.adjust_learning_rate(session, model, 
                                                      lr, config.lr_decay, train_history )

            if not os.path.exists(config.model_dir):
                print("Creating directory %s" % config.model_dir)
                os.mkdir(config.model_dir)

            if math.isnan(valid_mse):
                print("Training failed due to nan.")
                quit()
            elif stop_training(config,valid_history):
                print("Training stopped.")
                quit()
            else:
                if ( (config.early_stop is None) or 
                     (valid_history[-1] <= min(valid_history)) ):
                    model_utils.save_model(session,config,i)

#%%
def main(_):
    config = get_configs()

    train_model(config)

if __name__ == "__main__":
    tf.app.run()