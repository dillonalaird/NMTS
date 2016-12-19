from __future__ import division
from __future__ import print_function

import tensorflow as tf


def attention_luong(h_t, encoder_hs):
    with tf.variable_scope("attention_luong"):
        h_dim = h_t.get_shape().with_rank(2)[1].value

        Wc = tf.get_variable("Wc", shape=[2*h_dim, h_dim])
        bc = tf.get_variable("bc", shape=[h_dim])

        encoder_hs = tf.transpose(encoder_hs, perm=[1,0,2])
        scores = tf.reduce_sum(tf.mul(encoder_hs, h_t), 2)
        a_t    = tf.nn.softmax(tf.transpose(scores))
        a_t    = tf.expand_dims(a_t, 2)
        c_t    = tf.batch_matmul(tf.transpose(encoder_hs, perm=[1,2,0]), a_t)
        c_t    = tf.squeeze(c_t, [2])
        h_tld  = tf.tanh(tf.matmul(tf.concat(1, [h_t, c_t]), Wc) + bc)

    return h_tld


def attention_bahdanau(h_t, encoder_hs):
    pass


def attention_nmts(h_t, encoder_hs):
    with tf.variable_scope("attention_nmts"):
        h_dim = h_t.get_shape().with_rank(2)[1].value
        t_dim = encoder_hs.get_shape().with_rank(3)[1].value

        W1 = tf.get_variable("W1", shape=[2*h_dim, h_dim])
        b1 = tf.get_variable("b1", shape=[h_dim])
        W2 = tf.get_variable("W2", shape=[h_dim, 1])
        b2 = tf.get_variable("b2", shape=[1])

        s = []
        for h_tt in tf.split(1, t_dim, encoder_hs):
            h_tt = tf.squeeze(h_tt, [1])
            h = tf.nn.relu(tf.matmul(tf.concat(1, [h_t, h_tt]), W1) + b1)
            o = tf.matmul(h, W2) + b2
            s.append(tf.squeeze(o, [1]))

        s = tf.pack(s)
        p = tf.nn.softmax(tf.transpose(s))
        p = tf.expand_dims(p, 2)
        a = tf.batch_matmul(tf.transpose(encoder_hs, perm=[0,2,1]), p)
        a = tf.squeeze(a, [2])

    return a


def attention_nmts_fast(h_t, encoder_hs):
    with tf.variable_scope("attention_nmts_fast"):
        h_dim = h_t.get_shape().with_rank(2)[1].value
        t_dim = encoder_hs.get_shape().with_rank(3)[1].value
        encoder_hs = tf.pack([tf.concat(1, [tf.squeeze(h_tt, [1]), h_t])
                              for h_tt in tf.split(1, t_dim, encoder_hs)])
        encoder_hs = tf.transpose(encoder_hs, perm=[1, 0, 2])

        Wp = tf.get_variable("Wp", shape=[1, 1, 2*h_dim, h_dim])
        bp = tf.get_variable("bp", shape=[h_dim])
        encoder_hs = tf.reshape(encoder_hs, [-1, t_dim, 1, 2*h_dim])
        encoder_hs = tf.nn.conv2d(encoder_hs, Wp, [1, 1, 1, 1], "SAME") + bp
        encoder_hs = tf.reshape(encoder_hs, [-1, t_dim, h_dim])

        W1 = tf.get_variable("W1", shape=[1, 1, h_dim, h_dim])
        b1 = tf.get_variable("b1", shape=[h_dim])
        W2 = tf.get_variable("W2", shape=[1, 1, h_dim, 1])
        b2 = tf.get_variable("b2", shape=[1])

        h = tf.reshape(encoder_hs, [-1, t_dim, 1, h_dim])
        h = tf.nn.relu(tf.nn.conv2d(h, W1, [1, 1, 1, 1], "SAME") + b1)
        s = tf.squeeze(tf.nn.conv2d(h, W2, [1, 1, 1, 1], "SAME") + b2, [2,3])
        p = tf.nn.softmax(s)
        p = tf.expand_dims(p, 2)
        a = tf.batch_matmul(tf.transpose(encoder_hs, perm=[0,2,1]), p)
        a = tf.squeeze(a, [2])
    return a
