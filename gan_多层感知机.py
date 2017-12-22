# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 14:24:15 2017

@author: SalaFeng-
"""

import tensorflow as tf
import numpy as np
from skimage.io import imsave
import os
from tensorflow.examples.tutorials.mnist import input_data
batch_size =256
max_epoch =500
img_size =28 *28
z_size =100
h1_size =150
h2_size =300
learning_rate =0.0005
momentum =0.9
def build_generator(Z):
    w1 =tf.Variable(tf.truncated_normal([z_size,h1_size],stddev=0.1),name="g_w1",dtype=tf.float32)
    b1 =tf.Variable(tf.zeros([h1_size]),name="g_b1",dtype=tf.float32)
    h1 =tf.nn.relu(tf.matmul(Z,w1) +b1)
    w2 =tf.Variable(tf.truncated_normal([h1_size,h2_size],stddev=0.1),name="g_w2",dtype=tf.float32)
    b2 =tf.Variable(tf.zeros([h2_size]), name="g_b2", dtype=tf.float32)
    h2 =tf.nn.relu(tf.matmul(h1, w2) + b2)
    w3 =tf.Variable(tf.truncated_normal([h2_size,img_size],stddev=0.1),name="g_w3",dtype =tf.float32)
    b3 =tf.Variable(tf.zeros([img_size]),name="g_b3",dtype=tf.float32)
    h3 =tf.matmul(h2,w3)+b3
    x_generate = tf.nn.tanh(h3)
    g_params =[w1,b1,w2,b2,w3,b3]
    return x_generate,g_params
    
def build_discriminator(x_data,x_generator,keep_prob):
    x_in =tf.concat([x_data,x_generator],0)
    w1 = tf.Variable(tf.truncated_normal([img_size, h2_size], stddev=0.1), name="d_w1", dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([h2_size]), name="d_b1", dtype=tf.float32)
    h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x_in, w1) + b1), keep_prob)
    w2 = tf.Variable(tf.truncated_normal([h2_size, h1_size], stddev=0.1), name="d_w2", dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([h1_size]), name="d_b2", dtype=tf.float32)
    h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1, w2) + b2), keep_prob)
    w3 = tf.Variable(tf.truncated_normal([h1_size, 1], stddev=0.1), name="d_w3", dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([1]), name="d_b3", dtype=tf.float32)
    h3 = tf.matmul(h2, w3) + b3
    y_data =tf.nn.sigmoid(tf.slice(h3,[0,0],[batch_size,-1],name =None))
    y_generated = tf.nn.sigmoid(tf.slice(h3, [batch_size, 0], [-1, -1], name=None))
    d_params = [w1, b1, w2, b2, w3, b3]
    return y_data, y_generated, d_params

def show_result(batch_res, fname, grid_size=(8, 8), grid_pad=5):
    batch_res = 0.5 * batch_res.reshape((batch_res.shape[0], 28, 28)) + 0.5
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for i, res in enumerate(batch_res):
        if i >= grid_size[0] * grid_size[1]:
            break
        img = (res) * 255
        img = img.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w] = img
    imsave(fname, img_grid)

    
def train():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    x_data =tf.placeholder(tf.float32,[batch_size,img_size],name ="x_data")
    Z =tf.placeholder(tf.float32,[batch_size,z_size],name ="Z")
    keep_prob =tf.placeholder(tf.float32,name ="keep_prob")
    global_step =tf.Variable(0,name="global_step",trainable=False)
    
    x_generated,g_params =build_generator(Z)
    y_data, y_generated, d_params = build_discriminator(x_data, x_generated, keep_prob)
    
    d_loss = -tf.reduce_mean(tf.log(y_data) +tf.log(1-y_generated))
    g_loss = -tf.reduce_mean(tf.log(y_generated))
    
    #momentum 优化
    #optimizer =tf.train.MomentumOptimizer(learning_rate,momentum)
    #SGD 优化
    learning_rate=tf.train.exponential_decay(0.001,global_step,100,0.98,staircase=True)    
    optimizer=tf.train.GradientDescentOptimizer(learning_rate)


    d_trainer = optimizer.minimize(d_loss, var_list=d_params)
    g_trainer = optimizer.minimize(g_loss, var_list=g_params)
    #初始化
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    z_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
    
    steps = 50000 // batch_size
    for i in range(sess.run(global_step),max_epoch):
        for j in range(steps):
            x_value, _ = mnist.train.next_batch(batch_size)
            x_value = 2 * x_value.astype(np.float32) - 1
            z_value = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
            _,D_loss_curr=sess.run([d_trainer,d_loss],feed_dict ={x_data:x_value,Z:z_value,keep_prob: np.sum(0.7).astype(np.float32)})
            _,G_loss_curr=sess.run([g_trainer,g_loss],feed_dict={x_data: x_value,Z:z_value,keep_prob: np.sum(0.7).astype(np.float32)})         
        print('Epoch :{}  D_loss:{:0.4f} G_loss:{:0.4f}'.format(i,D_loss_curr,G_loss_curr))
        x_gen_val = sess.run(x_generated, feed_dict={Z: z_sample_val})
        path ="output/optimizer{}".format("SGD_expenential_decay")
        if not os.path.exists(path):
            os.mkdir(path)
        show_result(x_gen_val, "{}/{}-{:0.4f}.jpg".format(path,i,learning_rate))
        z_random_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
        x_gen_val = sess.run(x_generated, feed_dict={Z: z_random_sample_val})
        #show_result(x_gen_val, "output/out_lr={}_momentum={}/random_sample{}.jpg".format(learning_rate,momentum,i))
        sess.run(tf.assign(global_step, i + 1))
train()           
            
    
    