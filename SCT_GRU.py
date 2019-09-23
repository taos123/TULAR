# -*- coding: UTF-8 -*-

# Author : sun tao
# Time   : 2019/9/21
# E-mail : suntao@ict.ac.cn


import tensorflow as tf


def sct_cal(x, weights, biases, keep_prob, n_hidden, seq_length, attention):

    x = tf.transpose(x, [1, 0, 2])  # transform position for different dimension

    GRUCell = tf.nn.rnn_cell.GRUCell(n_hidden)

    # define dynamic rnn model
    GRUCcell = tf.nn.rnn_cell.DropoutWrapper(GRUCell, output_keep_prob=keep_prob)
    (outputs, states) = tf.nn.dynamic_rnn(GRUCell, x, time_major=True, seq_length=seq_length, dtype=tf.float32)

    new_outputs = attention(outputs, attention_size=300, time_major=True, return_alphas=False)

    val = tf.matmul(new_outputs, weights['out'])+biases['out']
    return val



