import os
import sys
from typing import List, Tuple
import numpy as np
import tensorflow as tf
from data_loader import Batch


class Model:
    def __init__(self, char_list: List[str],
                 ) -> None:
        self.char_list = char_list
        self.snap_ID = 0
        self.is_train = tf.compat.v1.placeholder(tf.bool, name='is_train')
        self.input_imgs = tf.compat.v1.placeholder(
            tf.float32, shape=(None, None, None))
        self.batches_trained = 0
        self.update_ops = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.compat.v1.train.AdamOptimizer().minimize(self.loss)
        self.sess, self.saver = self.setup_tf()
