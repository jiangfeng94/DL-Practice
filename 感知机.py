# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:31:30 2017

@author: SalaFeng-
"""

import tensorflow as tf

w =tf.Variable(tf.truncated_normal([2,1],-0.1,0.1))
b =tf.Variable(tf.truncated_normal([1],0.1))

x = tf.placeholder(tf.float32,shape =[None,2])
y = tf.placeholder(tf.float32,shape =[None,1])

output = tf.sigmoid(tf.matmul(x,w)+b)
cross_entropy =tf.reduce_mean(tf.square(output-y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess =tf.Session()
sess.run(tf.global_variables_initializer())
train_x =[[1.0,1.0],[0.0,0.0],[1.0,0.0],[0.0,1.0]]
train_y =[[1.0],[0.0],[0.0],[0.0]]

for i in range(1000):
    sess.run([train_step],feed_dict={x:train_x,y:train_y})
#测试
test_x = [[0.0,1.0],[0.0,0.0],[1.0,1.0],[1.0,0.0]]
print(sess.run(output, feed_dict={x:test_x}))
'''
[[ 0.12097029]
 [ 0.00320845]
 [ 0.8547312 ]
 [ 0.12096987]]

'''