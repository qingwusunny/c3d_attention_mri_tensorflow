# coding=utf-8


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_attline_shape = [150]
_att_shape = [-1, 5, 6, 5]

class ModelUnit(object):
	"""the units of CNN"""

	def __init__(self,
				 batch_norm_decay,
				 batch_norm_epsilon,
				 batch_size=1,
				 keep_prop=0.5,
				 alpha_line=0.01):
		self._batch_size = batch_size
		self._drop_rate = 1.0 - keep_prop
		# batch normalization
		self._batch_norm_decay = batch_norm_decay
		self._batch_norm_epsilon = batch_norm_epsilon
		# alpha_line
		self._alpha_line = alpha_line

	def _conv3d(self, x, weight, bias=None):
		x = tf.nn.conv3d(x, weight, [1,1,1,1,1], padding='SAME')
		if bias is None:
			return x
		else:
			return tf.nn.bias_add(x, bias)

	def _relu(self, x):
		return tf.nn.relu(x)

	def _batch_norm(self, x, is_training):
		return tf.contrib.layers.batch_norm(
			x,
			decay=self._batch_norm_decay,
			center=True,  # beta
			scale=True,  # gamma
			epsilon=self._batch_norm_epsilon,
			is_training=is_training,
			fused=True)#,
			#zero_debias_moving_mean=True)  # default is false

	def _max_pool3d(self, x, KernelSize=2, Stride=2):
		return tf.contrib.layers.max_pool3d(
			x,
			kernel_size=KernelSize,
			stride=Stride,
			padding='SAME')

	def _flatten(self, x):
		return tf.layers.flatten(x)

	def _fully_connect(self, x, weight, bias):
		return tf.matmul(x, weight) + bias

	def _dropout(self, x, is_training):
		return tf.layers.dropout(x, rate=self._drop_rate, training=is_training)

	def _attention_norm(self, x):
		# x -> [batch_sze,-1]
		x = self._flatten(x)
		# x = x-min, make all x>=0
		x_min = tf.reshape(tf.reduce_min(x, -1), [-1,1])
		x = x - x_min
		# x = x/sum, normalization
		x_sum = tf.reshape(tf.reduce_sum(x, -1), [-1,1])
		x = x / x_sum
		# if x < attention_line, then x = 0
		attention_line = tf.ones(_attline_shape, tf.float32) * self._alpha_line
		flag_comp = tf.to_float(tf.less_equal(attention_line, x))
		x = flag_comp * x
		# x -> shape
		x = tf.reshape(x, _att_shape)
		return x