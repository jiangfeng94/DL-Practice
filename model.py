# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:23:52 2018
@author: SalaFeng-
"""

import numpy as np
import tensorflow as tf

def discriminator(x, c, v, t):
    layer_c = c
    c_w1 = tf.Variable(tf.truncated_normal([843, 512], stddev=0.1), name="c_w1", dtype=tf.float32)
    c_b1 = tf.Variable(tf.zeros([512]), name="c_b1", dtype=tf.float32)
    layer_c = tf.nn.relu(tf.matmul(layer_c, c_w1) + c_b1)
    c_w2 = tf.Variable(tf.truncated_normal([512, 512], stddev=0.1), name="c_w2", dtype=tf.float32)        c_b2 = tf.Variable(tf.zeros([512]), name="c_b2", dtype=tf.float32)
    layer_c = tf.nn.relu(tf.matmul(layer_c, c_w2) + c_b2)
    layer_v = v
    v_w1 = tf.Variable(tf.truncated_normal([4, 512], stddev=0.1), name="v_w1", dtype=tf.float32)
    v_b1 = tf.Variable(tf.zeros([512]), name="v_b1", dtype=tf.float32)
    layer_c = tf.nn.relu(tf.matmul(layer_v, v_w1) + v_b1)
    v_w2 = tf.Variable(tf.truncated_normal([512, 512], stddev=0.1), name="v_w2", dtype=tf.float32)
    v_b2 = tf.Variable(tf.zeros([512]), name="v_b2", dtype=tf.float32)
    layer_v = tf.nn.relu(tf.matmul(layer_v, v_w2) + v_b2)

    layer_t = t
    t_w1 = tf.Variable(tf.truncated_normal([8, 512], stddev=0.1), name="t_w1", dtype=tf.float32)
    t_b1 = tf.Variable(tf.zeros([512]), name="t_b1", dtype=tf.float32)
    layer_t = tf.nn.relu(tf.matmul(layer_t, t_w1) + t_b1)
    t_w2 = tf.Variable(tf.truncated_normal([512, 512], stddev=0.1), name="t_w2", dtype=tf.float32)
    t_b2 = tf.Variable(tf.zeros([512]), name="t_b2", dtype=tf.float32)
    layer_t = tf.nn.relu(tf.matmul(layer_t, t_w2) + t_b2)

    layer_i = tf.concat([layer_c, layer_v, layer_t])
    i_w1 = tf.Variable(tf.truncated_normal([1536, 1024], stddev=0.1), name="i_w1", dtype=tf.float32)
    i_b1 = tf.Variable(tf.zeros([1024]), name="i_b1", dtype=tf.float32)
    layer_t = tf.nn.relu(tf.matmul(layer_i, i_w1) + i_b1)
    i_w2 = tf.Variable(tf.truncated_normal([1024, 1024], stddev=0.1), name="i_w2", dtype=tf.float32)
    i_b2 = tf.Variable(tf.zeros([1024]), name="i_b2", dtype=tf.float32)
    layer_i = tf.nn.relu(tf.matmul(layer_i, i_w2) + i_b2)

    layer_x =x
    x_w1 = tf.Variable(tf.random_normal([5, 5, 3, 64]))
    layer_x =tf.nn.conv2d(layer_x,x_w1,strides=2,padding="same")
    #weight_norm
    layer_x =tf.nn.dropout(layer_x,keep_prob=0.5)
    layer_x =tf.nn.relu(layer_x)

    x_w2 = tf.Variable(tf.random_normal([5, 5, 3, 64]))
    layer_x =tf.nn.conv2d(layer_x,x_w2,strides=2,padding="same")
    #weight_norm
    layer_x =tf.nn.dropout(layer_x,keep_prob=0.5)
    layer_x =tf.nn.relu(layer_x)
    x_w3 = tf.Variable(tf.random_normal([5, 5, 3, 128]))
    layer_x =tf.nn.conv2d(layer_x,x_w3,strides=2,padding="same")
    #weight_norm
    layer_x =tf.nn.dropout(layer_x,keep_prob=0.5)
    layer_x =tf.nn.relu(layer_x)

    x_w4 = tf.Variable(tf.random_normal([5, 5, 3, 256]))
    layer_x =tf.nn.conv2d(layer_x,x_w4,strides=2,padding="same")
    #weight_norm
    layer_x =tf.nn.dropout(layer_x,keep_prob=0.5)
    layer_x =tf.nn.relu(layer_x)

    layer_x =tf.reshape(layer_x, [-1])
    x_w5 = tf.Variable(tf.truncated_normal([256, 1024], stddev=0.1), name="x_w5", dtype=tf.float32)
    x_b5 = tf.Variable(tf.zeros([1024]), name="x_b5", dtype=tf.float32)
    layer_x = tf.nn.relu(tf.matmul(layer_x, x_w5) + x_b5)
    x_w6 = tf.Variable(tf.truncated_normal([1024, 1024], stddev=0.1), name="x_w6", dtype=tf.float32)
    x_b6 = tf.Variable(tf.zeros([1024]), name="x_b6", dtype=tf.float32)
    layer_x = tf.nn.relu(tf.matmul(layer_x, x_w6) + x_b6)

    layer =tf.concat(layer_x,layer_i)
    r_w1 =tf.Variable(tf.truncated_normal([2048, 1024], stddev=0.1), name="r_w1", dtype=tf.float32)
    r_b1 =tf.Variable(tf.zeros([1024]), name="r_b1", dtype=tf.float32)
    layer_r =tf.nn.relu(tf.matmul(layer, r_w1) + r_b1)
    r_w2 =tf.Variable(tf.truncated_normal([1024, 1024], stddev=0.1), name="r_w2", dtype=tf.float32)
    r_b2 =tf.Variable(tf.zeros([1024]), name="r_b2", dtype=tf.float32)
    layer_r =tf.nn.relu(tf.matmul(layer_r, r_w2) + r_b2)
    







