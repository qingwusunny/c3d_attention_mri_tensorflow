# coding=utf-8

'''
write .nii data and label to tfrecord
and split dataset to 10 dataset dataset0,dataset1,dataset2,dataset3,dataset4...,dataset9
len_dataset = [129, 129, 129, 130, 129, 129, 130, 129, 129, 130]
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import nibabel as nib
import numpy as np
import random
import copy


_filelist_name = 'source_file_ad_hc_rsmwc1.list'  # the file list return of convert_images_to_list
_org_dir = '/media/E/zhoujiao/tfrecord_rsmwc1_ad_hc/'  # the parent directory of tfrecord 


def preprocess(image_data):
	''' preprocess for image_data
	Args:
		image_data: return for nib.load(path).get_data()
	Retuens:
		preprocess for image_data with float32
	'''

	image_data[image_data<0] = 0
	data_mean = np.mean(image_data)
	data_std = np.std(image_data)
	image_data = (image_data - data_mean)/data_std
	image_data = np.float32(image_data)  # convert float64 to float32 for tf.float32

	return image_data


def get_data(filelist):
	''' read .nii data to nparray list
	Args:
		filelist: a list of str(image path and label)
	Returns:
		a list of nparray data
		a list of label
	'''

	data = []
	label = []
	for line in filelist:
		[path,lab] = line.strip('\n').split()
		image_data = nib.load(path).get_data()
		image_data = preprocess(image_data)
		data.append(image_data)
		lab = int(lab)
		if lab == 0 or lab == 2:
			label += [[1,0],[1,0],[1,0]]
			#path1 = path[:29] + 'noiseHC/noise15_' + path[38:]
			#path2 = path[:29] + 'noiseHC/noise10_' + path[38:]
			path1 = path[:29] + 'noiseAD/noise15_' + path[38:]
			path2 = path[:29] + 'noiseAD/noise10_' + path[38:]
		else:
			label += [[0,1],[0,1],[0,1]]
			#path1 = path[:29] + 'noiseMCI/noise15_' + path[39:]
			#path2 = path[:29] + 'noiseMCI/noise10_' + path[39:]
			path1 = path[:29] + 'noiseHC/noise15_' + path[38:]
			path2 = path[:29] + 'noiseHC/noise10_' + path[38:]
		image_data = nib.load(path1).get_data()
		image_data = preprocess(image_data)
		data.append(image_data)
		image_data = nib.load(path2).get_data()
		image_data = preprocess(image_data)
		data.append(image_data)

	return data, label

	
def mirror_transformation(image_data):
	'''mirror transformation for image_data
	Args:
		image_data: return for nib.load(path).get_data()
	Returns:
		image_data: preprocess for image_data with float32
		image_data2: mirror transformation for image_data  with float32
	'''
	image_data = preprocess(image_data)
	image_data2 = copy.deepcopy(image_data)
	for x in range(79):
		for y in range(95):
			for z in range(79):
				image_data2[78-x, y, z] = image_data[x, y, z]

	return image_data, image_data2


def get_data2(filelist):
	''' read .nii data to nparray list
	Args:
		filelist: a list of str(image path and label)
	Returns:
		a list of nparray data
		a list of label
	'''

	data = []
	label = []
	for line in filelist:
		[path,lab] = line.strip('\n').split()
		image_data = nib.load(path).get_data()
		image_data, image_data2 = mirror_transformation(image_data)
		data.append(image_data)
		data.append(image_data2)
		lab = int(lab)
		if lab == 0 or lab == 2:
			label += [[1,0],[1,0]]
		else:
			label += [[0,1],[0,1]]

	return data, label


def write_tfrecord(filelist, record_name):
	'''write filelist's data to TFRecord
	Args:
		filelist: a list of str(image path and label)
		record_name: to write TFRecord file's name 
	'''

	tfrecord_filename = _org_dir + record_name + '.tfrecord'
	writer = tf.python_io.TFRecordWriter(tfrecord_filename)
	data, label = get_data2(filelist)
	print(len(data), len(label))
	for i in range(len(label)):
		tfrecord_features = {}
		tfrecord_features['feature'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[data[i].tostring()]))
		tfrecord_features['label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=label[i]))
		exmp = tf.train.Example(features=tf.train.Features(feature=tfrecord_features))
		writer.write(exmp.SerializeToString())
	writer.close()


def get_filename():
	f = open(_filelist_name, 'r')
	lines = list(f)
	[path, lab] = lines[0].strip('\n').split(' ')
	print(path)


def write_alldata_tfrecord():
	'''split all data to evaluation dataset, test dataset, train dataset and write them to tfrecord
	'''

	f = open(_filelist_name, 'r')
	lines = list(f)
	random.shuffle(lines)
	data_split = [int(0.1*i*len(lines)) for i in range(11)]
	for i in range(10):
		sub_lines = lines[data_split[i]:data_split[i+1]]
		write_tfrecord(sub_lines, 'dataset_'+str(i))
		print('%d:%d'%(i,len(sub_lines)))

	
if __name__ == '__main__':
	write_alldata_tfrecord()
	# get_filename()