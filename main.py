from __future__ import division
from __future__ import print_function

from nmts import NMTS
from data import unique_tokens

import os
import pprint
import cPickle as pkl
import tensorflow as tf


pp = pprint.PrettyPrinter().pprint

flags = tf.app.flags

flags.DEFINE_integer("h_dim", 512, "The dimension of the hidden state [512]")
flags.DEFINE_integer("num_layers", 2, "The number of NMTS layers, must be 2 or more [2]")
flags.DEFINE_integer("epochs", 12, "The number of epochs [12]")
flags.DEFINE_integer("batch_size", 128, "The batch size [128]")
flags.DEFINE_integer("max_length", 40, "Maximum length of sentences [30]")
flags.DEFINE_integer("beam", 5, "Beam size for beam search [5]")
flags.DEFINE_float("learning_rate", 1.0, "The learning rate [1.0]")
flags.DEFINE_float("gradient_clip", 5.0, "Value to clip gradients at [5.0]")
flags.DEFINE_float("alpha", 0.017, "Weight of maximum likelihood loss [0.017]")
flags.DEFINE_boolean("is_test", False, "Run algorithm on test set [False]")
flags.DEFINE_string("optimizer", "SGD", "Type of optimizer to use [SGD]")
flags.DEFINE_string("attention", "luong", "What type of attention to use [luong]")
flags.DEFINE_string("dataset", "debug", "Dataset to use [debug]")
flags.DEFINE_string("name", "default", "name of model [default]")
flags.DEFINE_string("sample", None, "Paoth of sample file [None]")

FLAGS = flags.FLAGS


class debug:
    source_data_path       = "data/train.debug.source.en.tokenized"
    target_data_path       = "data/train.debug.target.fr.tokenized"
    valid_source_data_path = "data/train.debug.source.en.tokenized"
    valid_target_data_path = "data/train.debug.target.fr.tokenized"
    test_source_data_path  = "data/train.debug.source.en.tokenized"
    test_target_data_path  = "data/train.debug.target.fr.tokenized"
    vocab_path             = "data/small.vocab.pkl"


class small:
    source_data_path       = "data/train.small.source.en.tokenized"
    target_data_path       = "data/train.small.target.fr.tokenized"
    valid_source_data_path = "data/valid.small.source.en.tokenized"
    valid_target_data_path = "data/valid.small.target.fr.tokenized"
    vocab_path             = "data/small.vocab.pkl"


class large:
    source_data_path       = "data/train.source.en.en.tokenized"
    target_data_path       = "data/train.target.fr.de.tokenized"
    valid_source_data_path = "data/valid.source.en.en.tokenized"
    valid_target_data_path = "data/valid.target.fr.de.tokenized"
    test_source_data_path  = "data/test.source.en.en.tokenized"
    test_target_data_path  = "data/test.target.fr.de.tokenized"
    vocab_path             = "data/vocab.pkl"



def get_model_dir(config, exceptions):
    attrs = config.__dict__["__flags"]
    keys = list(attrs.keys())
    keys.sort()

    names = ["{}={}".format(key, attrs[key]) for key in keys if key not in exceptions]
    model_dir = os.path.join(*names)
    ckpt_dir = os.path.join("checkpoints", model_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    return model_dir


def print_samples(samples):
    for sample in samples:
        for s in sample:
            if s == "</s>":
                break
            print(" " + s, end="")
        print()


def main(_):
    config = FLAGS
    if config.dataset == "debug":
        data_config = debug
    elif config.dataset == "small":
        data_config = small
    elif config.dataset == "large":
        data_config = large
    else:
        raise ValueError("Unrecognized dataset: {}".format(config.dataset))

    if config.num_layers < 2:
        raise ValueError("Number of layers must be 2 or greater")

    with open(data_config.vocab_path, "rb") as f:
        vocab = pkl.load(f)

    vocab_size = len(vocab) + len(unique_tokens)
    config.s_nwords = vocab_size
    config.t_nwords = vocab_size

    pp(config.__dict__["__flags"])
    config.vocab      = vocab
    config.inv_vocab  = {v:k for k,v in vocab.iteritems()}
    with tf.Session() as sess:
        nmts = NMTS(config, sess, get_model_dir(config,
            ["batch_size", "s_nwords", "t_nwords", "vocab", "inv_vocab", "sample",
             "is_test", "beam"]))
        if config.sample:
            nmts.load()
            # TODO: using batch_size for beam size
            samples = nmts.beam_search(config.sample, 5)
            print_samples(samples)
        else:
            if config.is_test:
                bleus, bleus_cum, lens, total_eval = nmts.get_bleu_score(
                                    config.beam,
                                    data_config.test_source_data_path,
                                    data_config.test_target_data_path)
                for bleu, bleu_cum, l, num_eval in zip(bleus, bleus_cum, lens, total_eval):
                    print("[Test] [Bleu: {}] [Bleu Cum.: {}] [Len.: {}] [Eval: {}]"
                          .format(bleu, bleu_cum, l, num_eval))
            else:
                nmts.run(config.epochs, config.beam,
                         data_config.source_data_path,
                         data_config.target_data_path,
                         data_config.valid_source_data_path,
                         data_config.valid_target_data_path)


if __name__ == "__main__":
    tf.app.run()
