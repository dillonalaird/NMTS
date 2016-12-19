from __future__ import division
from __future__ import print_function

from rnn_cell import NMTSDecoderCellOld
from rnn_cell import MultiSkipRNNCell
from attention import attention_luong
from attention import attention_nmts_fast
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops

import tensorflow as tf


def nmts_encoder(num_layers, x_dim, h_dim, inputs, sequence_length=None):
    with tf.variable_scope("nmts_encoder"):
        emb = tf.get_variable("emb", shape=[x_dim, h_dim])
        inputs = tf.nn.embedding_lookup(emb, inputs)

        t_dim = inputs.get_shape().with_rank(3)[1].value

        Wp = tf.get_variable("Wp", shape=[1, 1, 2*h_dim, h_dim])
        bp = tf.get_variable("bp", shape=[h_dim])

        cellf, cellb = tf.nn.rnn_cell.LSTMCell(h_dim), tf.nn.rnn_cell.LSTMCell(h_dim)

        outputs, state_fb = tf.nn.bidirectional_dynamic_rnn(cellf, cellb, inputs,
                                                            sequence_length=sequence_length,
                                                            dtype=tf.float32,
                                                            swap_memory=True)
        outputs = tf.concat(2, outputs)
        # hack to multiply W*h_t over time
        outputs = tf.reshape(outputs, [-1, t_dim, 1, 2*h_dim])
        outputs = tf.nn.conv2d(outputs, Wp, [1, 1, 1, 1], "SAME")
        outputs = tf.squeeze(outputs, [2]) + bp

        cell = tf.nn.rnn_cell.LSTMCell(h_dim)
        multi_cell = MultiSkipRNNCell([cell]*(num_layers - 1))

        outputs, states = tf.nn.dynamic_rnn(multi_cell, outputs,
                                            sequence_length=sequence_length,
                                            dtype=tf.float32, swap_memory=True)
    return outputs, state_fb + states


def nmts_states_projection(states):
    with tf.variable_scope("nmts_states_projection"):
        h_dim = states[0].h.get_shape().with_rank(2)[1].value

        Wc = tf.get_variable("Wc", shape=[2*h_dim, h_dim])
        bc = tf.get_variable("bc", shape=[h_dim])
        Wh = tf.get_variable("Wh", shape=[2*h_dim, h_dim])
        bh = tf.get_variable("bh", shape=[h_dim])

        c = tf.matmul(tf.concat(1, [states[0].c, states[1].c]), Wc) + bc
        h = tf.matmul(tf.concat(1, [states[0].h, states[1].h]), Wh) + bh

        state0 = tf.nn.rnn_cell.LSTMStateTuple(c, h)
    return tuple([state0] + list(states[2:]))


def nmts_decoder_attention_old(num_layers, x_dim, h_dim, o_dim, inputs, encoder_hs,
                               attention="luong", feed_previous=False,
                               initial_state=None):
    with tf.variable_scope("nmts_decoder") as decoder_scope:
        emb = tf.get_variable("emb", shape=[x_dim, h_dim])
        inputs = tf.nn.embedding_lookup(emb, inputs)

        batch_size = inputs.get_shape().with_rank(3)[0].value
        max_length = inputs.get_shape().with_rank(3)[1].value

        Wo = tf.get_variable("Wo", [h_dim, o_dim])
        bo = tf.get_variable("bo", [o_dim])

        cell = tf.nn.rnn_cell.LSTMCell(h_dim)
        multi_cell = NMTSDecoderCellOld([cell]*(num_layers - 1))

        if initial_state:
            state = initial_state[0]
            states = initial_state[1:]
        else:
            state = cell.zero_state(batch_size, tf.float32)
            states = multi_cell.zero_state(batch_size, tf.float32)

        all_outputs = []
        logits = []
        all_inputs = tf.split(1, max_length, inputs)
        for t in xrange(max_length):
            if t > 0: decoder_scope.reuse_variables()
            if not feed_previous or t == 0:
                input_t = tf.squeeze(all_inputs[t], [1])
            output, state = cell(input_t, state)
            if attention == "luong":
                c_t = attention_luong(output, encoder_hs)
            elif attention == "nmts":
                c_t = attention_nmts_fast(output, encoder_hs)
            else:
                raise ValueError("Unknown attention: {}".format(attention))
            output, states = multi_cell(output, (states, c_t))
            states, _ = states
            all_outputs.append(output)

            logit = tf.matmul(output, Wo) + bo
            logits.append(logit)

            if feed_previous:
                input_t = tf.cast(tf.argmax(logit, 1), tf.int32)
                input_t = tf.nn.embedding_lookup(emb, input_t)

    return all_outputs, logits


def get_batch_size(inputs):
    first_input = inputs
    while nest.is_sequence(first_input):
        first_input = first_input[0]

    if first_input.get_shape().ndims != 1:
        input_shape = first_input.get_shape().with_rank_at_least(2)
        fixed_batch_size = input_shape[0]

        flat_inputs = nest.flatten(inputs)
        for flat_input in flat_inputs:
            input_shape = flat_input.get_shape().with_rank_at_least(2)
            batch_size, input_size = input_shape[0], input_shape[1:]
            fixed_batch_size.merge_with(batch_size)
            for i, size in enumerate(input_size):
                if size.value is None:
                    raise ValueError("Input size (dimension {} of inputs) must be "
                            "accessible via shape inference, but saw value None."
                            .format(i))
    else:
        fixed_batch_size = first_input.get_shape().with_rank_at_least(1)[0]

    if fixed_batch_size.value:
        batch_size = fixed_batch_size.value
    else:
        batch_size = array_ops.shape(first_input)[0]

    return batch_size


def nmts_decoder_attention(cell, x_dim, h_dim, o_dim, inputs, encoder_hs,
                           sequence_length=None, initial_state=None):
    with tf.variable_scope("nmts_decoder"):
        emb = tf.get_variable("emb", shape=[x_dim, h_dim])
        inputs = tf.nn.embedding_lookup(emb, inputs)

        batch_size = inputs.get_shape().with_rank(3)[0].value
        max_length = inputs.get_shape().with_rank(3)[1].value

        Wo = tf.get_variable("Wo", [1, 1, h_dim, o_dim])
        bo = tf.get_variable("bo", [o_dim])

        if not initial_state:
            initial_state = cell.zero_state(batch_size, tf.float32)
        output, state = tf.nn.dynamic_rnn(cell, inputs,
                                          sequence_length=sequence_length,
                                          initial_state=(initial_state, encoder_hs),
                                          swap_memory=True)
        output = tf.reshape(output, [-1, max_length, 1, h_dim])
        output = tf.nn.conv2d(output, Wo, [1, 1, 1, 1], "SAME")
        output = tf.squeeze(output, [2]) + bo

        return output, state
