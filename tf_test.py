# -*- coding: UTF-8 -*-

# Author : sun tao
# Time   : 2019/9/5
# E-mail : suntao@ict.ac.cn

import tensorflow as tf

inputs = tf.Variable(tf.random_normal([1, 3, 4]))
attention_size = 5
hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

# Trainable parameters
w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

w = tf.Variable(tf.random_normal([attention_size, attention_size], stddev=0.1))

w1 = tf.Variable(tf.random_normal([attention_size, attention_size], stddev=0.1))
w2 = tf.Variable(tf.random_normal([attention_size, attention_size], stddev=0.1))

with tf.name_scope('v'):
    # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
    #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
    v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

# For each of the timestamps its vector of size A from `v` is reduced with `u` vector

# vu for score
# vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape

# vwu for score
vw = tf.tensordot(v, w, axes=1, name='vu')  # (B,T,A) shape
vwu = tf.tensordot(vw, u_omega, axes=1, name="vwu")  # (B,T) shape
alphas = tf.nn.softmax(vwu, name='alphas')  # (B,T) shape

# vtanh(w1ht+w2hs) for score


# Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

print(sess.run(inputs))
print(sess.run(w_omega))
print(sess.run(b_omega))
print(sess.run(u_omega))
print(sess.run(tf.tensordot(inputs, w_omega, axes=1)))
print(sess.run(tf.tensordot(inputs, w_omega, axes=1) + b_omega))
print(sess.run(tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)))

print(sess.run(tf.tensordot(v, u_omega, axes=1, name='vu')))


