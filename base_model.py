# -*- coding: utf-8 -*-
"""
Created on Thu May 16 10:48:06 2019

@author: user
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class BaseModel(object):

    def train_step(self, sess, batch, keep_prob=1.0):

        feed_dict = self._get_feed_dict(batch,keep_prob=keep_prob,training=True)

        (mse, _) = sess.run([self._mse,self._train_op],feed_dict)

        return mse

    def step(self, sess, batch):

        feed_dict = self._get_feed_dict(batch,keep_prob=1.0,training=False)

        (mse, preds) = sess.run([self._mse,self._predictions],feed_dict)

        return mse, preds

    def debug_step(self, sess, batch, training=False):
        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=3)

        feed_dict = self._get_feed_dict(batch,keep_prob=1.0,training=training)

        (mse, preds) = sess.run([self._mse,self._predictions], feed_dict)
        return mse

    def _get_feed_dict(self, batch, keep_prob=1.0, training=False):
        feed_dict = dict()

        feed_dict[self._batch_size] = batch.inputs[0].shape[0]
        feed_dict[self._seq_lengths] = batch.seq_lengths
        feed_dict[self._keep_prob] = keep_prob
        feed_dict[self._phase] = 1 if training is True else 0

        for i in range(self._max_unrollings):
            feed_dict[self._inputs[i]]  = batch.inputs[i]
            feed_dict[self._targets[i]] = batch.targets[i]

        return feed_dict

    def _center_and_scale(self, x):
        n = tf.shape(x)[1]
        return tf.divide( x - self._center[:n], self._scale[:n] )

    def _reverse_center_and_scale(self, x):
        n = tf.shape(x)[1]
        return tf.multiply( x, self._scale[:n] ) + self._center[:n]

    def set_scaling_params(self,session,center=None,scale=None):
        assert(center is not None)
        assert(scale is not None)
        session.run(tf.assign(self._center,center))
        session.run(tf.assign(self._scale,scale))

    def set_learning_rate(self, session, lr_value):
        session.run(tf.assign(self._lr, lr_value))
        return lr_value

    @property
    def inputs(self):
        return self._inputs

    @property
    def targets(self):
        return self._targets

    @property
    def cost(self):
        return self._cost

    @property
    def lr(self):
        return self._lr

    @property
    def max_unrollings(self):
        return self._max_unrollings
