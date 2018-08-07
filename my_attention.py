# coding=utf-8


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import model_unit


_initial_path = '/home/tf-1/zhoujiao/c3d-caffemodel/'  # the parent directory of weights and biases
_weight_decay = 0.0001
_fc_weight_decay = 0.001
_batch_norm_decay = 0.9  # decay for the moving average, small for avoid over-fitting
_batch_norm_epsilon = 0.001  # small float added to variance to avoid dividing by zero.
_batch_size = 5
_keep_prop = 0.5
_alpha_line = 0.01
_num_classes = 2
Unit = model_unit.ModelUnit(_batch_norm_decay, _batch_norm_epsilon, _batch_size, _keep_prop, _alpha_line)


def _variable_on_cpu(name, shape=None, init_name=None, is_bias=False):
	''' create a variable stored on CPU memory
	Args:
		name: name of the variable
		shape: list of ints, shape of variable
			   if use .npy to initial the variable, then shape=None
		init_name: .npy name of value used to initializer the variable
				   if random initial the variable, then init_name=None
		is_bias: .npy is bias or weights
	Returns:
		var: variable tensor
	'''

	with tf.device('/cpu:0'):
		dtype = tf.float32
		if init_name is None:
			if is_bias:
				init = tf.constant_initializer(0.1)
			else:
				init = tf.contrib.layers.xavier_initializer(uniform=False)
		else:
			init_data = np.load(_initial_path + init_name)
			if is_bias:
				init = init_data.reshape([init_data.shape[4]])
			else:
				init = init_data.transpose([2,3,4,1,0])
		var = tf.get_variable(name, shape, initializer=init, dtype=dtype)
	return var


def _variable_with_regularization(name, shape=None, init_name=None, weight_decay=_weight_decay):
	''' create a variable with weight decay
	Args:
		name: name of the variable
		shape: list of ints, shape of variable
			   if use .npy to initial the variable, then shape=None
		init_name: .npy name of value used to initializer the variable
				   if random initial the variable, then init_name=None 
		weight_decay: the factor of variable's L2 loss
					  if don't need the variable's L2 loss, weight_decay=None
	Returns:
		var: variable tensor
	'''
	
	var = _variable_on_cpu(name, shape, init_name)
	if weight_decay is not None:
		l2_loss = tf.multiply(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
		tf.add_to_collection('losses', l2_loss)
	return var


def inference(images, is_training):
	''' build a metric model
		Args:
			images: returned from image process
			is_training: training or testing
		Returns:
			Logits
	'''

	# conv1
	with tf.variable_scope('conv1') as scope:
		weight = _variable_with_regularization('weights', init_name='conv1a_weight.npy')
		bias = _variable_on_cpu('biases',init_name='conv1a_bias.npy', is_bias=True)
		x = Unit._conv3d(images, weight, bias)
		x = Unit._relu(x)
		x = Unit._max_pool3d(x)
	# conv2
	with tf.variable_scope('conv2') as scope:
		weight = _variable_with_regularization('weights', init_name='conv2a_weight.npy')
		bias = _variable_on_cpu('biases', init_name='conv2a_bias.npy', is_bias=True)
		x = Unit._conv3d(x, weight, bias)
		x = Unit._relu(x)
		x = Unit._max_pool3d(x)	
	# conv3a
	with tf.variable_scope('conv3a') as scope:
		weight = _variable_with_regularization('weights', init_name='conv3a_weight.npy')
		x = Unit._conv3d(x, weight)
		x = Unit._batch_norm(x, is_training)
		x = Unit._relu(x)
	# conv3b
	with tf.variable_scope('conv3b') as scope:
		weight = _variable_with_regularization('weights', init_name='conv3b_weight.npy')
		bias = _variable_on_cpu('biases', init_name='conv3b_bias.npy', is_bias=True)
		x = Unit._conv3d(x, weight, bias)
		x = Unit._relu(x)
		x = Unit._max_pool3d(x)	

	# get attention
	with tf.variable_scope('atten_conv1a') as scope:
		weight = _variable_with_regularization('weights', shape=[3,3,3,256,128], weight_decay=None)	
		alpha = Unit._conv3d(x, weight)
		alpha = Unit._batch_norm(alpha, is_training)
		alpha = Unit._relu(alpha)
	with tf.variable_scope('atten_conv1b') as scope:
		weight = _variable_with_regularization('weights', shape=[3,3,3,128,64], weight_decay=None)
		bias = _variable_on_cpu('biases', shape=[64])
		alpha = Unit._conv3d(alpha, weight, bias)
		alpha = Unit._relu(alpha)
		alpha = Unit._max_pool3d(alpha)
	with tf.variable_scope('atten_conv2') as scope:
		weight = _variable_with_regularization('weights', shape=[1,1,1,64,1], weight_decay=None)
		bias = _variable_on_cpu('biases', shape=[1])
		alpha = Unit._conv3d(alpha, weight, bias)
	with tf.name_scope('attention_norm') as scope:
		alpha = Unit._attention_norm(alpha)

	# conv4a
	with tf.variable_scope('conv4a') as scope:
		weight = _variable_with_regularization('weights', init_name='conv4a_weight.npy')
		x = Unit._conv3d(x, weight)
		x = Unit._batch_norm(x, is_training)
		x = Unit._relu(x)
	# conv4b 
	with tf.variable_scope('conv4b') as scope:
		weight = _variable_with_regularization('weights', init_name='conv4b_weight.npy')
		bias = _variable_on_cpu('biases', init_name='conv4b_bias.npy', is_bias=True)
		x = Unit._conv3d(x, weight,bias)
		x = Unit._relu(x)
		x = Unit._max_pool3d(x)	
	# attention op
	with tf.name_scope('attention_op') as scope:
		x = tf.einsum('ijklm,ijkl->ijklm', x, alpha)
	# conv5a
	with tf.variable_scope('conv5a') as scope:
		weight = _variable_with_regularization('weights', init_name='conv5a_weight.npy')
		x = Unit._conv3d(x, weight)
		x = Unit._batch_norm(x, is_training)
		x = Unit._relu(x)
	# conv5b
	with tf.variable_scope('conv5b') as scope:
		weight = _variable_with_regularization('weights', init_name='conv5b_weight.npy')
		bias = _variable_on_cpu('biases', init_name='conv5b_bias.npy', is_bias=True)
		x = Unit._conv3d(x, weight, bias)
		x = Unit._relu(x)
		x = Unit._max_pool3d(x)	
	# fc1
	with tf.variable_scope('fc1') as scope:
		weight= _variable_with_regularization('weigths', shape=[3*3*3*512,4096], weight_decay=_fc_weight_decay)
		bias = _variable_on_cpu('biases', shape=[4096])
		x = Unit._flatten(x)
		x = Unit._fully_connect(x, weight, bias)
		x = Unit._relu(x)
		x = Unit._dropout(x, is_training)
	# fc2
	with tf.variable_scope('fc2') as scope:
		weight = _variable_with_regularization('weights', shape=[4096,2048], weight_decay=_fc_weight_decay)
		bias = _variable_on_cpu('biases', shape=[2048])
		x = Unit._fully_connect(x, weight, bias)
		x = Unit._relu(x)
		x = Unit._dropout(x, is_training)
	# fc3
	with tf.variable_scope('fc3') as scope: 
		weight = _variable_with_regularization('weights', shape=[2048,_num_classes], weight_decay=_fc_weight_decay)
		bias = _variable_on_cpu('biases', shape=[_num_classes])
		x = Unit._fully_connect(x, weight, bias)

	return x	
	

def loss(logits, labels):
	'''add l2loss and cross entropy
	Args:
		logits: returned from inference()
		labels: labels of inputs
	Returns:
		loss tensor of type float
	'''

	cross_entropy = tf.reduce_mean(
					tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits))
	tf.add_to_collection('losses', cross_entropy)

	return tf.add_n(tf.get_collection('losses'), name='total_loss')


def predict(logits):
	pred = tf.argmax(logits, axis=-1)
	return pred


