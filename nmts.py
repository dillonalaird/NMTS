from __future__ import division
from __future__ import print_function

from datetime import datetime
from bleu.length_analysis import process_files
from rnn import nmts_encoder
from rnn import nmts_decoder_attention
from rnn import nmts_states_projection
from rnn_cell import NMTSDecoderCell
from data import data_iterator_wp
from data import _PAD
from data import _BOS
from data import _EOS
from data import _UNK

import tensorflow as tf
import numpy as np
import time
import sys
import os


class NMTS(object):
    def __init__(self, config, sess, model_dir):
        self.h_dim      = config.h_dim
        self.num_layers = config.num_layers
        self.lr         = config.learning_rate
        self.g_clip     = config.gradient_clip
        self.s_nwords   = config.s_nwords
        self.t_nwords   = config.t_nwords
        self.alpha      = config.alpha
        self.batch_size = config.batch_size
        self.max_length = config.max_length
        self.attention  = config.attention
        self.vocab      = config.vocab
        self.inv_vocab  = config.inv_vocab
        self.optimizer  = config.optimizer
        self.name       = config.name
        self.is_test    = config.is_test
        self.sess       = sess
        self.model_dir  = model_dir

        self.train_iters = 0

        self.x = tf.placeholder(tf.int32, [None, self.max_length], name="x")
        self.y = tf.placeholder(tf.int32, [None, self.max_length], name="y")
        self.x_len = tf.placeholder(tf.int32, [None], name="x_len")
        self.y_len = tf.placeholder(tf.int32, [None], name="y_len")

        initializer = tf.random_normal_initializer(0.0, stddev=0.01)
        with tf.variable_scope("NMTS", initializer=initializer):
            self.build_model()

    def build_model(self):
        self.encoder_hs, state = nmts_encoder(self.num_layers, self.s_nwords,
                                             self.h_dim, self.x, sequence_length=self.x_len)
        cell = NMTSDecoderCell([tf.nn.rnn_cell.LSTMCell(self.h_dim)]*self.num_layers,
                               attention=self.attention)
        self.initial_state = nmts_states_projection(state)
        logits, state = nmts_decoder_attention(cell, self.t_nwords, self.h_dim,
                                               self.t_nwords, self.y, self.encoder_hs,
                                               sequence_length=self.y_len,
                                               initial_state=self.initial_state)
        self.decoder_state = state[0]

        # log softmax
        self.log_probs = tf.nn.log_softmax(logits)
        logits = [tf.squeeze(logit, [1]) for logit in
                  tf.split(1, self.max_length, logits)][:-1]
        targets = tf.split(1, self.max_length, self.y)[1:]
        weights = tf.unpack(tf.sequence_mask(self.y_len - 1, self.max_length - 1,
                                             dtype=tf.float32), None, 1)
        probs = tf.exp(self.log_probs)
        gleu = self.gleu_score(logits, targets)
        loss_rl = self.loss_rl(probs, gleu)
        loss_ml = tf.nn.seq2seq.sequence_loss(logits, targets, weights)
        self.loss = self.alpha*loss_ml + loss_rl
        self.optim = tf.contrib.layers.optimize_loss(self.loss, None,
                self.lr, self.optimizer, clip_gradients=self.g_clip,
                summaries=["learning_rate", "loss", "gradient_norm"])
        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver()

    def loss_rl(self, probs, glue):
        max_probs = tf.reduce_max(probs, [2])
        loss_rl = tf.reduce_sum([glue*tf.squeeze(prob, [1]) for prob in
                                 tf.split(1, self.max_length, max_probs)][:-1], 0)
        batch_size = tf.cast(tf.shape(loss_rl)[0], tf.float32)
        return tf.reduce_sum(loss_rl, 0)/batch_size

    def gleu_score(self, logits, truth, N=4):
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
        return tf.cast(tf.reduce_sum(equals, 0), tf.float32)/equals.get_shape()[0].value

    def _stack_state(self, state, order, beam):
        new_state = []
        for layer in state:
            c = layer.c[order,:]
            h = layer.h[order,:]
            new_state.append(tf.nn.rnn_cell.LSTMStateTuple(c, h))

        return tuple(new_state)

    def _copy_first_state(self, state, beam):
        new_state = []
        for layer in state:
            c = np.tile(layer.c[0,:], (beam, 1))
            h = np.tile(layer.h[0,:], (beam, 1))
            new_state.append(tf.nn.rnn_cell.LSTMStateTuple(c, h))

        return tuple(new_state)

    def beam_search(self, source_data_path, beam):
        iterator = data_iterator_wp(source_data_path, source_data_path,
                                    self.vocab, self.max_length, 1)

        samples = []
        for dsource, slen, _, _ in iterator:
            beam_probs = []
            beam_idxs  = []
            for i in xrange(self.max_length):
                if i == 0:
                    dtarget   = [[_BOS] + [_PAD]*(self.max_length - 1)]
                    feed_dict = {self.x: dsource, self.x_len: slen, self.y: dtarget,
                                 self.y_len: [1]}
                    log_probs, encoder_hs, state = self.sess.run([self.log_probs,
                                                                 self.encoder_hs,
                                                                 self.decoder_state],
                                                                 feed_dict=feed_dict)
                    log_probs  = log_probs[0,0,:]
                    topk       = np.argsort(log_probs)[-beam:]
                    beam_probs = [log_probs[j] for j in topk]
                    beam_idxs  = [[top] for top in topk]
                    y          = [[top] + [_PAD]*(self.max_length - 1) for top in topk]
                    state      = self._copy_first_state(state, beam)
                    encoder_hs = np.tile(encoder_hs, (beam, 1, 1))
                else:
                    feed_dict = {self.y: y, self.y_len: [1]*beam,
                                 self.encoder_hs: encoder_hs,
                                 self.initial_state: state}
                    log_probs, state = self.sess.run([self.log_probs, self.decoder_state],
                                                     feed_dict=feed_dict)


                    log_probs = log_probs[:beam,0,:]
                    next_log_probs = []
                    for i, log_prob in enumerate(log_probs):
                        next_log_probs.append(beam_probs[i] + log_prob)
                    next_log_probs = np.array(next_log_probs)

                    topk = np.dstack(np.unravel_index(np.argsort(next_log_probs.ravel()),
                        next_log_probs.shape))[0][-beam:,:]
                    next_beam_idxs  = []
                    next_beam_probs = []
                    y = []
                    for idxs in topk:
                        next_beam_idxs.append(beam_idxs[idxs[0]] + [idxs[1]])
                        next_beam_probs.append(beam_probs[idxs[0]] + log_probs[idxs[0],idxs[1]])
                        y.append([idxs[1]] + [_PAD]*(self.max_length - 1))
                    beam_idxs  = next_beam_idxs
                    beam_probs = next_beam_probs
                    state      = self._stack_state(state, topk[:,0], beam)

                if all([idxs[-1] == _EOS for idxs in beam_idxs]):
                    break

            best_beam = beam_idxs[np.argmax(beam_probs)]
            sample = []
            for idx in best_beam:
                if idx == _BOS:
                    s = "<s>"
                elif idx == _EOS:
                    s = "</s>"
                elif idx == _PAD:
                    s = "<pad>"
                elif idx == _UNK:
                    s = "<unk>"
                else:
                    s = self.inv_vocab[idx]
                sample.append(s)
            samples.append(sample)

        return samples

    def sample(self, source_data_path):
        iterator = data_iterator_wp(source_data_path, source_data_path,
                                    self.vocab, self.max_length,
                                    self.batch_size)
        y       = None
        samples = []
        for dsource, slen, _, _ in iterator:
            batch_samples = []
            for i in xrange(self.max_length):
                if i == 0:
                    dtarget = [[_BOS] + [_PAD]*(self.max_length - 1)]
                    dtarget *= self.batch_size
                    tlen    = [1]*self.batch_size
                    feed_dict = {self.x: dsource, self.x_len: slen,
                                 self.y: dtarget, self.y_len: tlen}
                    log_probs, encoder_hs, state = self.sess.run([self.log_probs,
                                                                 self.encoder_hs,
                                                                 self.decoder_state],
                                                                 feed_dict=feed_dict)
                else:
                    tlen      = [1]*self.batch_size
                    feed_dict = {self.y: y, self.y_len: tlen,
                                 self.encoder_hs: encoder_hs,
                                 self.initial_state: state}
                    log_probs, state = self.sess.run([self.log_probs, self.decoder_state],
                                                     feed_dict=feed_dict)

                y = []
                log_probs = log_probs[:,0,:]
                for j, log_prob in enumerate(log_probs):
                    idx = np.argmax(log_prob)
                    if idx == _BOS:
                        s = "<s>"
                    elif idx == _EOS:
                        s = "</s>"
                    elif idx == _PAD:
                        s = "<pad>"
                    elif idx == _UNK:
                        s = "<unk>"
                    else:
                        s = self.inv_vocab[idx]

                    if i == 0:
                        batch_samples.append([s])
                    else:
                        batch_samples[j].append(s)
                    y.append([idx] + [_PAD]*(self.max_length - 1))

            samples.extend(batch_samples)

        return samples

    def train(self, epoch, beam, source_data_path, target_data_path, merged_sum,
              writer, valid_source_data_path, valid_target_data_path):
        iterator = data_iterator_wp(source_data_path, target_data_path,
                                    self.vocab, self.max_length,
                                    self.batch_size)
        i = 0
        total_loss = 0.
        #best_bleu  = 0.
        best_valid = float("inf")
        for dsource, slen, dtarget, tlen in iterator:
            outputs = self.sess.run([self.loss, self.optim, merged_sum],
                                    feed_dict={self.x: dsource,
                                               self.x_len: slen,
                                               self.y: dtarget,
                                               self.y_len: tlen})
            loss = outputs[0]
            itr = self.train_iters*epoch + i
            total_loss += loss
            if itr % 10 == 0:
                writer.add_summary(outputs[-1], itr)
            if itr % 10 == 0:
                print("[Train] [Time: {}] [Epoch: {}] [Iteration: {}] [Loss: {}]"
                      .format(datetime.now(), epoch, itr, loss))
                sys.stdout.flush()
            if itr > 0 and itr % 1000 == 0:
                #bleus, bleus_cum, lens, total_eval = self.get_bleu_score(beam,
                #                                                         valid_source_data_path,
                #                                                         valid_target_data_path)
                #for bleu, bleu_cum, l, num_eval in zip(bleus, bleus_cum, lens, total_eval):
                #    print("[Valid] [Bleu: {}] [Bleu Cum.: {}] [Len.: {}] [Eval: {}]"
                #          .format(bleu, bleu_cum, l, num_eval))
                #sys.stdout.flush()
                #if bleu_cum > best_bleu:
                #    best_bleu = bleu_cum
                #    self.saver.save(self.sess, os.path.join("checkpoints",
                #                                            self.model_dir,
                #                                            self.name + ".bestbleu"),
                #                    global_step=itr)
                valid_loss = self.test(valid_source_data_path, valid_target_data_path)
                print("[Valid] [Loss: {}]".format(valid_loss))
                if valid_loss < best_valid:
                    best_valid = valid_loss
                    self.saver.save(self.sess, os.path.join("checkpoints",
                                                            self.model_dir,
                                                            self.name + ".bestvalid"),
                                    global_step=itr)

            i += 1
        self.train_iters = i
        return total_loss/i

    def test(self, source_data_path, target_data_path):
        iterator = data_iterator_wp(source_data_path, target_data_path,
                                    self.vocab, self.max_length,
                                    self.batch_size)
        i = 0
        total_loss = 0.
        for dsource, slen, dtarget, tlen in iterator:
            loss, = self.sess.run([self.loss],
                                  feed_dict={self.x: dsource,
                                             self.x_len: slen,
                                             self.y: dtarget,
                                             self.y_len: tlen})
            total_loss += loss
            i += 1

        return total_loss/i

    def get_bleu_score(self, beam, source_data_path, target_data_path):
        samples = self.beam_search(source_data_path, beam)
        hyp_file = "hyp" + str(int(time.time()))
        max_l = 0
        with open(hyp_file, "wb") as f:
            for sample in samples:
                for i,s in enumerate(sample):
                    if s == "</s>":
                        break
                    if i > max_l: max_l = i
                    f.write(" " + s)
                f.write("\n")

        if max_l == 0:
            os.remove(hyp_file)
            return [0], [0], [0], [0]

        bleus, bleus_cum, lens, total_eval = process_files(hyp_file,
                                                           target_data_path,
                                                           self.max_length)
        os.remove(hyp_file)
        return bleus, bleus_cum, lens, total_eval

    def run(self, epochs, beam, source_data_path, target_data_path,
            valid_source_data_path, valid_target_data_path):
        merged_sum = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("./logs/{}".format(self.model_dir),
                                        self.sess.graph)
        #best_valid_loss = float("inf")
        for epoch in xrange(epochs):
            train_loss = self.train(epoch, beam, source_data_path,
                                    target_data_path, merged_sum, writer,
                                    valid_source_data_path,
                                    valid_target_data_path)
            #valid_loss = self.test(valid_source_data_path, valid_target_data_path)
            #if epoch % 100 == 0:
            #    bleus, bleus_cum, lens, total_eval = self.get_bleu_score(beam,
            #                                                             valid_source_data_path,
            #                                                             valid_target_data_path)
            #    for bleu, bleu_cum, l, num_eval in zip(bleus, bleus_cum, lens, total_eval):
            #        print("[Valid] [Bleu: {}] [Bleu Cum.: {}] [Len.: {}] [Eval: {}]"
            #              .format(bleu, bleu_cum, l, num_eval))
            print("[Train] [Avg. Loss: {}]".format(train_loss))
            #print("[Valid] [Loss: {}]".format(valid_loss))
            sys.stdout.flush()
            #if valid_loss < best_valid_loss:
            #    best_valid_loss = valid_loss
            #    self.saver.save(self.sess, os.path.join("checkpoints", self.model_dir,
            #                                            self.name + ".bestvalid"),
            #                    global_step=epoch)

    def load(self):
        print("[*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(os.path.join("checkpoints",
                                                          self.model_dir))
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise Exception("[!] No checkpoint found")
