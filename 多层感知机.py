# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 13:04:41 2017

@author: SalaFeng-
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist =input_data.read_data_sets('MNIST_data',one_hot=True)

sess =tf.InteractiveSession()

in_units =784

h1_units =300  #隐藏层 节点数

#初始化参数
# mean是均值 stddev是标准差
w1 = tf.Variable(tf.truncated_normal([in_units,h1_units],stddev = 0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
w2 = tf.Variable(tf.zeros([h1_units,10]))
b2 = tf.Variable(tf.zeros([10]))
x =tf.placeholder(tf.float32,[None,in_units])
keep_prob = tf.placeholder(tf.float32)

#隐含层计算
hidden1 = tf.nn.relu(tf.add(tf.matmul(x,w1),b1))
#随即将一部分节点设为0
hidden_drop = tf.nn.dropout(hidden1,keep_prob)
"""
drop_out 防止过拟合，一般用于全连接层，卷积部分一般不会用到drop_out
keep_prob: 神经元被选中的概率
在初始化时keep_prob 一般是一个占位符  keep_prob = tf.placeholder(tf.float32)tensorflow在run时设置keep_prob具体的值，例如keep_prob: 0.5

"""
#输出层计算
y = tf.nn.softmax(tf.matmul(hidden_drop,w2)+b2)

y_ = tf.placeholder(tf.float32,[None,10])
#计算损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))

train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
#初始化所有变量
tf.global_variables_initializer().run()

for i in range(3000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys,keep_prob:0.75})
    

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
#cast(x, dtype, name=None)    将x的数据格式转化成dtype.
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))