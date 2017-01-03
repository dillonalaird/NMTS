from __future__ import division
from __future__ import print_function

import tensorflow as tf


def gleu_score_basic(logits, truth, N=4):
    # TODO: not accounting for actual prediction/target lengths
    predictions = [tf.cast(tf.argmax(logit, 1), tf.int32) for logit in logits]
    truth       = [tf.squeeze(t, [1]) for t in truth]
    p_ngrams = []
    t_ngrams = []
    for n in xrange(1, N + 1):
        p_ngrams.append([predictions[i:i+n] for i in xrange(len(predictions) - n + 1)])
        t_ngrams.append([truth[i:i+n] for i in xrange(len(truth) - n + 1)])

    equals = []
    for n in xrange(N):
        equal = tf.equal(p_ngrams[n], t_ngrams[n])
        # foldl initializes array of True
        equal = tf.foldl(lambda acc, x: tf.logical_and(acc, x),
                         tf.split(1, n + 1, equal))
        equal = tf.cast(tf.squeeze(equal, [1]), tf.int32)
        equals.append(equal)

    equals = tf.concat(0, equals)
    return tf.cast(tf.reduce_sum(equals, 0), tf.float32)/equals.get_shape()[0].value, equals


def gleu_score(logits, truth, N=4):
    # TODO: not accounting for actual prediction/target lengths
    predictions = [tf.cast(tf.argmax(logit, 1), tf.int32) for logit in logits]
    truth       = [tf.squeeze(t, [1]) for t in truth]
    p_ngrams = []
    t_ngrams = []
    for n in xrange(1, N + 1):
        p_ngrams.append([predictions[i:i+n] for i in xrange(len(predictions) - n + 1)])
        t_ngrams.append([truth[i:i+n] for i in xrange(len(truth) - n + 1)])

    #equals = []
    #for n in xrange(N):
    #    for i, p_ngram in enumerate(p_ngrams[n]):
    #        equal = tf.foldl(lambda acc, x: tf.logical_or(acc, tf.equal(x, p_ngram)),
    #                         t_ngrams[n],
    #                         initializer=tf.cast(tf.zeros_like(p_ngram), tf.bool))
    #        equal = tf.foldl(lambda acc, x: tf.logical_and(acc, x),
    #                         tf.split(0, n + 1, equal))
    #        equal = tf.cast(tf.squeeze(equal, [0]), tf.int32)
    #        equals.append(equal)

    equals = []
    for n in [2]:
        tmps = []
        for i, p_ngram in enumerate(p_ngrams[n]):
            equal = tf.foldl(lambda acc, x: tf.logical_or(acc, tf.equal(x, p_ngram)),
                             t_ngrams[n],
                             initializer=tf.cast(tf.zeros_like(p_ngram), tf.bool))
            tmps.append(equal)
            equal = tf.foldl(lambda acc, x: tf.logical_and(acc, x),
                             tf.split(0, n + 1, equal))
            equal = tf.cast(tf.squeeze(equal, [0]), tf.int32)
            equals.append(equal)

    equals = tf.pack(equals)
    return (tf.cast(tf.reduce_sum(equals, 0), tf.float32)/equals.get_shape()[0].value,
            equals, p_ngrams[2], t_ngrams[2], tmps)
