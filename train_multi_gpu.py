# coding=utf-8


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import time
import os
from datetime import datetime

import my_attention as metric_model
# import my_c3d as metric_model

from my_attention import _batch_size 


# os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'

parser = argparse.ArgumentParser()

parser.add_argument('--num_gpus', type=list, default='0',
				   help='a list of gpu number')
parser.add_argument('--max_step', type=int, default=50000,
				   help='number of training iterations')

FLAGS = parser.parse_args()


_num_epochs_per_decay = 25
_initial_learning_rate = 0.0001
_lerning_rate_decay_factor = 0.1
_tower_name = 'tower'
_moving_average_decay = 0.9999
_test_interval = 100
_max_step = FLAGS.max_step
_train_dir = '/media/E/zhoujiao/models/mycode_2018.4/swmc1_ad_hc'
_log_device_placement = False  # print the log of device placement or not
_train_dataset_num = [3,4,5,6,7,8,9]
_test_dataset_num = [0,1,2]
# _dataset_len = [129, 129, 129, 129, 129, 129, 129, 129, 129, 129]  # ad_hc
# _dataset_len = [129, 132, 132, 132, 132, 129, 132, 132, 132, 132]  # hc_mci
_dataset_len = [86, 86, 86, 86, 86, 86, 86, 86, 86, 86]  # rswmc1 ad_hc
_image_shape = [79,95,79,1]
# _parent_dir = '/media/E/zhoujiao/tfrecord_noise_ad_hc/'
_parent_dir = '/media/E/zhoujiao/tfrecord_rsmwc1_ad_hc/'


gpu_name = list(map(int,FLAGS.num_gpus))
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in gpu_name)

def tower_loss(scope, logits, labels):
	'''calculate the total loss on a single tower
	Args:
		scope: the identity of tower
		images: the input 5D tensor of shape [batch_size, 79, 95, 79, 1]
		logits: the output of metric_model.inference()
		labels: the label 1D tensor of shape [batch_size]
	Returns:
		the total loss for the tower
	'''

	#logits = metric_model.inference(images, is_training)

	_ = metric_model.loss(logits, labels)

	losses = tf.get_collection('losses', scope)

	total_loss = tf.add_n(losses, name='total_loss')

	return total_loss


def average_gradients(tower_grads):
	'''calculate the average gradient for each shared variable across all towers
	Args:
		tower_grads:list of lists of (gradient, variable) tuples.
					The outer list is over individual gradients.
					The inner list is over the gradient calculation for each tower.
	Returns:
		list of pairs of (gradient, variable) where the gradient has been averaged across all towers.
	'''

	average_grads = []
	for grad_and_vars in zip(*tower_grads):
		# the grad_and_vars is:((gard0_gpu0, var0_gpu0),...,(grad0_gpuN, var0_gpuN))
		grads = []
		for g, _ in grad_and_vars:
			expended_g = tf.expand_dims(g,0)
			grads.append(expended_g)
		grad = tf.concat(axis=0, values=grads)
		grad = tf.reduce_mean(grad,0)
	
		# keep in mind that the variables are redundant because they are shared across towers
		# so we will just return the first tower's pointer to the variables
		v = grad_and_vars[0][1]
		grad_and_var = (grad,v)
		average_grads.append(grad_and_var)

	return average_grads


def accuracy(logits, lables):
	correct_pred = tf.equal(metric_model.predict(logits), tf.argmax(lables, axis=-1))
	acc_list = tf.cast(correct_pred, tf.float32)
	acc = tf.reduce_mean(acc_list)

	return acc


def parse_exmp(serial_exmp):
	feats = tf.parse_single_example(serial_exmp, features={'feature':tf.FixedLenFeature([], tf.string),
															'label':tf.FixedLenFeature([2], tf.int64)})
	image = tf.decode_raw(feats['feature'], tf.float32)
	image = tf.reshape(image, _image_shape)
	lable = feats['label']
	
	return image, lable


def train():
	with tf.Graph().as_default(), tf.device('/cpu:0'):
		handle = tf.placeholder(tf.string, [])
		is_training = tf.placeholder(tf.bool)

		train_dataset = tf.data.TFRecordDataset(_parent_dir + 'dataset_'+str(_train_dataset_num[0])+'.tfrecord')
		num_train = _dataset_len[_train_dataset_num[0]]
		for i in _train_dataset_num[1:]:
			train_dataset = train_dataset.concatenate(tf.data.TFRecordDataset(_parent_dir + 'dataset_' + str(i) + '.tfrecord'))
			num_train += _dataset_len[i]
		test_dataset = tf.data.TFRecordDataset(_parent_dir + 'dataset_'+str(_test_dataset_num[0])+'.tfrecord')
		num_test = _dataset_len[_test_dataset_num[0]]
		for i in _test_dataset_num[1:]:
			test_dataset = train_dataset.concatenate(tf.data.TFRecordDataset(_parent_dir + 'dataset_' + str(i) + '.tfrecord'))
			num_test += _dataset_len[i]
		train_dataset = train_dataset.map(parse_exmp)
		# train_dataset = train_dataset.shuffle(1000).batch(_batch_size).repeat(_epoches)
		train_dataset = train_dataset.shuffle(1000).batch(_batch_size).repeat()
		test_dataset = test_dataset.map(parse_exmp)
		test_dataset = test_dataset.batch(1).repeat()
		train_iter = train_dataset.make_one_shot_iterator()
		test_iter = test_dataset.make_one_shot_iterator()
		iter = tf.data.Iterator.from_string_handle(handle, train_iter.output_types)
		

		num_batches_per_epoch = num_train//_batch_size
		global_step  = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)  # 
		decay_steps = int(num_batches_per_epoch * _num_epochs_per_decay)

		lr  = tf.train.exponential_decay(_initial_learning_rate, global_step, decay_steps, _lerning_rate_decay_factor, staircase=True)

		# opt = tf.train.GradientDescentOptimizer(lr)
		opt = tf.train.AdamOptimizer(lr)

		tower_grads = []
		acc_list = []
		with tf.variable_scope(tf.get_variable_scope()):
			for i in gpu_name:
				with tf.device('/gpu:%d' % i):
					with tf.name_scope('%s_%d' % (_tower_name, i)) as scope:
						images,labels = iter.get_next()
						images = tf.concat([images,images,images], axis=-1)
						logits = metric_model.inference(images, is_training)
						loss = tower_loss(scope, logits, labels)
						acc_list +=  [accuracy(logits, labels)]
						# reuse variables for the next tower
						tf.get_variable_scope().reuse_variables()
						# calculate the gradients for the batch of data on the tower
						grads = opt.compute_gradients(loss)
						# keep track of the gradients across all towers
						tower_grads.append(grads)

		acc = sum(acc_list)/len(acc_list)
		grads = average_gradients(tower_grads)


		apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)  # make the global_step +1

		# track the moving averages of all trainable variables
		variable_averages = tf.train.ExponentialMovingAverage(_moving_average_decay, global_step)
		variables_averages_op = variable_averages.apply(tf.trainable_variables())

		# group all updates to into a single train op
		train_op = tf.group(apply_gradient_op, variables_averages_op)
		# train_op = apply_gradient_op

		# create a saver
		saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)

		init  = tf.global_variables_initializer()

		# start running operations on the Graph
		# allow_soft_placement must be set to True to build towers on GPU, as some of the ops do not have GPU implementations
		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=_log_device_placement))
		train_handle = sess.run(train_iter.string_handle())
		test_handle = sess.run(test_iter.string_handle())
		sess.run(init)


		for step in range(_max_step):
			start_time = time.time()

			if step > 999 and step % 50 == 0:
				duration = time.time() - start_time
				num_examples_per_step = _batch_size * len(gpu_name)
				examples_per_sec = num_examples_per_step / duration
				sec_per_batch = duration /len(gpu_name)
				if step % 200 == 0:
					_, loss_value, train_acc = sess.run([train_op, loss, acc], feed_dict={handle: train_handle, is_training: True})
					assert not np.isnan(loss_value), 'Model diverged with loss=Nan'
					format_str = ('%s: step %d, loss = %.3f accurcy = %.3f(%.1f examples/sec; %.3f sec/batch)')
					print(format_str % (datetime.now(), step, loss_value, train_acc, examples_per_sec, sec_per_batch))
				else:
					_, loss_value = sess.run([train_op, loss], feed_dict={handle: train_handle, is_training: True})
					assert not np.isnan(loss_value), 'Model diverged with loss=Nan'
					format_str = ('%s: step %d, loss = %.3f(%.1f examples/sec; %.3f sec/batch)')
					print(format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))
			else:
				_, loss_value = sess.run([train_op, loss], feed_dict={handle: train_handle, is_training: True})
				assert not np.isnan(loss_value), 'Model diverged with loss=Nan'


			if step > 999 and step % _test_interval == 0:
				test_acclist = []
				for i in range(num_test//(len(gpu_name))):
					# newacclist, p, l= sess.run([acc_list,logits,labels], feed_dict={handle: test_handle, is_training: False})
					newacclist= sess.run(acc_list, feed_dict={handle: test_handle, is_training: False})
					test_acclist += newacclist
					# print(p,l)
				print('step=', step, ' len_acc=', len(test_acclist), 'test_acc = ', sum(test_acclist)/len(test_acclist))
				# if sum(test_acclist)/len(test_acclist)>=0.80:
					# checkpoint_path = os.path.join(_train_dir, 'model.ckpt')
					# saver.save(sess, checkpoint_path, global_step = step)

			# if step % 1000 == 0 or (step + 1) == max_step:
				# checkpoint_path = os.path.join(_train_dir, 'model.ckpt')
				# saver.save(sess, checkpoint_path, global_step=step)
		sess.close()


def main(argv=None):
	train()


if __name__ == '__main__':
	tf.app.run()		