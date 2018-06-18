#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/7 16:18
# @Author  : Zehan Song
# @Site    :
# @File    : cnn.py
# @Software: PyCharm

# set your config class
import tensorflow as tf
import warnings
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#默认为0：输出所有log信息
#设置为1：进一步屏蔽INFO信息
#设置为2：进一步屏蔽WARNING信息
#设置为3：进一步屏蔽ERROR信息
class Config(object):
	# data_path parameters
	class_path = "/home/szh/TIP/scene_classes.csv"
	data_path = "/home/szh/AIchallenger/baseline/final/"
	train_data_json = "/home/szh/TIP/scene_train_annotations_20170904.json"
	val_data_json = "/home/szh/TIP/scene_validation_annotations_20170908.json"
	#train parameters
	lr = 0.001
	batch_size = 64
	max_epoch = 50
	#evaluation parameters
	save_path = "/home/szh/TIP/model_path"
	evaluation_period = 10

def parse(self,kwargs):
	for k,v in kwargs.items():
		if not hasattr(self,k):
			warnings.warn("your config doesn't have that key(%s)" %(k))
		else:
			setattr(self,k,v)
	print('user config:')
	print('#################################')
	for k in dir(self):
		if not k.startswith('_') and k != 'parse' and k != 'state_dict':
			print k, getattr(self, k)
	print('#################################')
	return self

def state_dict(self):
	return {k: getattr(self, k) for k in dir(self) if not k.startswith('_') and k != 'parse' and k != 'state_dict'}

Config.parse = parse
Config.state_dict = state_dict
opt = Config()

#set your batcher
import json
import pandas as pd
import numpy as np
import random
from PIL import Image
import cv2
class_names = pd.read_csv(opt.class_path,header=None)
number_ids = class_names[0]
chinese_names = class_names[1]
english_names = class_names[2]


class Batcher(object):
	def __init__(self,train,data_path,json_path,batch_size=opt.batch_size):
		with open(json_path,'r') as f:
			data = json.load(f)
		self.train = train
		self.batch_size = batch_size
		self.all_img_paths = [os.path.join(opt.data_path,d["image_id"]) for d in data]
		self.all_label_ids = [d["label_id"] for d in data]
		self.start = 0
		self.data_index = list(range(len(self.all_img_paths)))
		if self.train:
			random.shuffle(self.data_index)
	def reset(self):
		self.start = 0
		if self.train:
			random.shuffle(self.data_index)
	def img_resize(self, imgpath, img_size):
		# resize the image to the specific size
		img = Image.open(imgpath)

		if (img.width > img.height):
			scale = float(img_size) / float(img.height)
			img = np.array(cv2.resize(np.array(img), (
			int(img.width * scale + 1), img_size))).astype(np.float32)
		else:
			scale = float(img_size) / float(img.width)
			img = np.array(cv2.resize(np.array(img), (
			img_size, int(img.height * scale + 1)))).astype(np.float32)
		# crop the proper size and scale to [-1, 1]
		img = (img[
				  (img.shape[0] - img_size) // 2:
				  (img.shape[0] - img_size) // 2 + img_size,
				  (img.shape[1] - img_size) // 2:
				  (img.shape[1] - img_size) // 2 + img_size,
				  :]-127)/255
		return img
	def get_data(self):
		# begin = time.time()
		if not self.train:
			self.batch_size = len(self.all_img_paths)

		# config = tf.ConfigProto()
		# config.gpu_options.allow_growth = True
		# config.allow_soft_placement = True
		# sess = tf.Session(config=config)
		image_batch = []
		label_batch = []
		if self.start>len(self.all_img_paths):
			self.reset()
		self.end = min(self.start+self.batch_size,len(self.all_img_paths))
		for i in range(self.start,self.end):
			img = self.img_resize(self.all_img_paths[i],128)
			# img = tf.image.convert_image_dtype(img,dtype=tf.float32)
			# img = tf.image.resize_images(img,[128,128])
			# if self.train:
			#     img = tf.image.random_flip_up_down(img)
			#     img = tf.image.random_flip_left_right(img)
			#     img = tf.image.random_brightness(img,0.1)
			# img = sess.run(img)
			#u can find other methods to implement data augmentation like some predefined functions
			#of cv2. I don't want to use functions wrt tf because a session would be neccessary adding
			#extra much more overhead of time

			image_batch.append(img)
			label_batch.append(self.all_label_ids[i])
		self.start += self.batch_size
		# print("start")
		# print self.start
		# sess.close()
		# end = time.time()
		# print("use time")
		# print(end-begin)
		# print np.array(image_batch).size
		return np.array(image_batch),np.array(label_batch)


#set your model
class Model(object):
	def weight_variable(self,shape, stddev=0.1):
		initial = tf.truncated_normal(shape, stddev=stddev)
		return tf.Variable(initial)
	def bias_variable(self,shape, bais=0.1):
		initial = tf.constant(bais, shape=shape)
		return tf.Variable(initial)
	def conv2d(self,x, w):
		return tf.nn.conv2d(x, w, [1, 1, 1, 1], 'SAME')
	def max_pool_2x2(self,x):
		return tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
	def max_pool_3x3(self,x):
		return tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
	def avg_pool_3x3(self,x):
		return tf.nn.avg_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
	def bulid_model(self):
		# network structure
		# conv1
		# self.train = tf.placeholder(tf.bool)
		self.features = tf.placeholder(dtype=tf.float32, shape=(None,128, 128, 3))
		# self.features = tf.cond(self.train,lambda: self.preprocess(self.rawimage),lambda: self.nopreprocess(self.rawimage))
		# self.features = tf.placeholder(dtype=tf.float32,shape=(None,128,128,3),name='features-input')
		self.labels = tf.placeholder(dtype=tf.int32,shape=(None),name='labels-input')
		# self.labels = tf.one_hot(indices=self.labels,depth=80)
		W_conv1 = self.weight_variable([5, 5, 3, 64], stddev=1e-4)
		b_conv1 = self.bias_variable([64])
		h_conv1 = tf.nn.relu(self.conv2d(self.features, W_conv1) + b_conv1)
		h_pool1 = self.max_pool_3x3(h_conv1)
		# norm1
		norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
		# conv2
		W_conv2 = self.weight_variable([5, 5, 64, 64], stddev=1e-2)
		b_conv2 = self.bias_variable([64])
		h_conv2 = tf.nn.relu(self.conv2d(norm1, W_conv2) + b_conv2)
		# norm2
		norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
		h_pool2 = self.max_pool_3x3(norm2)
		# conv3
		W_conv3 = self.weight_variable([5, 5, 64, 64], stddev=1e-2)
		b_conv3 = self.bias_variable([64])
		h_conv3 = tf.nn.relu(self.conv2d(h_pool2, W_conv3) + b_conv3)
		h_pool3 = self.max_pool_3x3(h_conv3)
		# fc1
		W_fc1 = self.weight_variable([16 * 16 * 64, 128])
		b_fc1 = self.bias_variable([128])
		h_pool3_flat = tf.reshape(h_pool3, [-1, 16 * 16 * 64])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
		# introduce dropout
		self.keep_prob = tf.placeholder("float")
		h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
		# fc2
		W_fc2 = self.weight_variable([128, 80])
		b_fc2 = self.bias_variable([80])
		self.y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#set train and validation process
from tqdm import tqdm
class Trainer(object):
	def __init__(self,train_batcher,val_batcher,model):
		self.train_batcher = train_batcher
		self.val_batcher = val_batcher
		self.model = model
		self.model.bulid_model()
		self.saver = tf.train.Saver()

	def train(self):
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.allow_soft_placement = True
		self.sess = tf.Session(config=config)
		cross_entropy_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits\
												(labels=self.model.labels,logits=self.model.y_conv))
		variables = tf.trainable_variables()
		regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in variables])
		loss = cross_entropy_mean + regularization_cost
		# one_hot_labels = tf.one_hot(indices=self.model.labels,depth=80,dtype=tf.int32)
		accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k\
					(predictions=self.model.y_conv,\
					 targets=self.model.labels,k=1)\
						   ,tf.float32))
		optimizer = tf.train.AdamOptimizer(opt.lr)
		train_op = optimizer.minimize(loss)
		TRAINING_STEPS = len(self.train_batcher.all_img_paths)/opt.batch_size
		self.sess.run(tf.global_variables_initializer())
		best_accuracy = 0
		for epoch in range(opt.max_epoch):
			total_loss = 0
			total_accuracy = 0
			for i in tqdm(range(TRAINING_STEPS)):
				# begin = time.time()
				imgs,labels = self.train_batcher.get_data()
				feed = {self.model.features:imgs,self.model.labels:labels,\
						self.model.keep_prob:0.5}
				_,loss_value,accuracy_value = self.sess.run([train_op,loss,accuracy],feed_dict=feed)
				total_loss += loss_value
				total_accuracy += accuracy_value
				# end = time.time()
				# print("train use time")
				# print(end-begin)
				if i%10 == 0:
					print("iter[%d/%d]:loss=%.6f,accuracy=%.6f" % (i+1, TRAINING_STEPS, \
																	loss_value, \
																	accuracy_value))
			print("epoch[%d/%d]:loss=%.6f,accuracy=%.6f"%(epoch+1,opt.max_epoch,\
														  total_loss/TRAINING_STEPS,\
														  total_accuracy/TRAINING_STEPS))
			if epoch%opt.evaluation_period==0:
				imgs,labels = self.val_batcher.get_data()
				feed = {self.model.features: imgs, self.model.labels: labels, \
						self.model.keep_prob: 1.0}
				loss_value, accuracy_value = self.sess.run([loss, accuracy], feed_dict=feed)
				print("val_loss = %.6f, val_accuracy = %.6f"%\
					  (loss_value,accuracy_value))
				if best_accuracy<accuracy_value:
					best_accuracy = accuracy_value
					self.saver.save(self.sess,"bestmodel.ckpt")

		'''if u have the ground truths of test dataset,the u can write like that'''
		# img, labels = self.test_batcher.get_data()
		# feed = {self.model.features: img, self.model.labels: labels, \
		#         self.model.keep_prob: 1.0}
		# loss_value, accuracy_value = self.sess.run([loss, accuracy], feed_dict=feed)
		self.sess.close()

def main(**kwargs):
	opt.parse(kwargs)
	train_batcher = Batcher(train=True, data_path=opt.data_path, json_path=opt.train_data_json, \
							batch_size=opt.batch_size)
	val_batcher = Batcher(train=False, data_path=opt.data_path, json_path=opt.val_data_json)
	model = Model()
	trainer = Trainer(train_batcher,val_batcher,model)
	trainer.train()

if __name__ == "__main__":
	import fire
	fire.Fire()