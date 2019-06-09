#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow import layers as tf_layers

class CNN_MNIST_MODEL(object):
    def __init__(self,input_x,input_y,lr,lr_delay,decay_steps):
        self._input_x = input_x
        self._input_y = input_y
        self._lr = lr
        self._decay_steps = decay_steps
        self.lr_delay = lr_delay
    def forward(self):
        self.model = tf_layers.conv2d(self._input_x,16,(3,3),(1,1),"Same",activation=tf.nn.relu)
        self.model = tf_layers.max_pooling2d(self.model,(2,2),(2,2),"Same")
        self.model = tf_layers.conv2d(self.model,32,(3,3),(1,1),"Same",activation=tf.nn.relu)
        self.model = tf_layers.max_pooling2d(self.model,(2,2),(2,2),"Same")
        self.model = tf_layers.flatten(self.model)
        self.model = tf_layers.dense(self.model,500,activation=tf.nn.tanh)
        self.model = tf_layers.dense(self.model,10,activation=tf.nn.softmax)
        return self.model
    def train(self):
        loss = tf.reduce_mean(tf.reduce_sum(-self._input_y * tf.log(self.model)))
        global_step = tf.Variable(0,trainable=False)
        learning_rate = tf.train.exponential_decay(self._lr,global_step,self._decay_steps,self.lr_delay,False)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step)
        
        #acc
        real_index = tf.argmax(self._input_y,axis=1)
        pre_index = tf.argmax(self.model,axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(real_index,pre_index),tf.float32))
        
        return train_op,loss,learning_rate,global_step,accuracy    

