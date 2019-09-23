# -*- coding: UTF-8 -*-

# Author : sun tao
# Time   : 2019/9/21
# E-mail : suntao@ict.ac.cn


import tensorflow as tf


def sct_cal(x, weights, biases, keep_prob, n_hidden, seq_length, attention):

    x = tf.transpose(x, [1, 0, 2])  # transform position for different dimension

    fw_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=0.5, state_is_tuple=True)
    fw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(fw_lstm_cell, output_keep_prob=keep_prob)  # use dropout
    bw_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=0.5, state_is_tuple=True)
    bw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(bw_lstm_cell, output_keep_prob=keep_prob)

    # bidirectional LSTM
    cell_fw = tf.nn.rnn_cell.MultiRNNCell([fw_lstm_cell], state_is_tuple=True)
    cell_bw = tf.nn.rnn_cell.MultiRNNCell([bw_lstm_cell], state_is_tuple=True)
    # dynamic rnn
    (outputs, states) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x, dtype=tf.float32, time_major=True, sequence_length=seq_length)  # ,dtype=tf.float32,time_major=True  ,initial_state_fw=istate_fw,initial_state_bw=istate_bw

    new_outputs = attention(outputs, attention_size=300, time_major=True, return_alphas=False)

    val = tf.matmul(new_outputs, weights['out'])+biases['out']
    return val



