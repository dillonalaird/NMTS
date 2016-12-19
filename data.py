from __future__ import division
from __future__ import print_function

from random import shuffle


_PAD = 0
_BOS = 1
_EOS = 2
_UNK = 3

unique_tokens = [_PAD, _BOS, _EOS, _UNK]
# Assumes that these tokens are the last 5 characters of the line, excluding the
# newline character
language_tokens = {"<2en>", "<2fr>", "<2de>"}


def pre_pad(lst, pad_elt, max_len):
    nlst = [pad_elt]*max_len
    nlst[(max_len - len(lst)):] = lst
    return nlst


def post_pad(lst, pad_elt, max_len):
    nlst = [pad_elt]*max_len
    nlst[:len(lst)] = lst
    return nlst


def read_vocabulary(data_path):
    return {w:i for i,w in enumerate(open(data_path).read().splitlines())}


def data_iterator_wp(source_data_path,
                     target_data_path,
                     vocab,
                     max_length,
                     batch_size):
    with open(source_data_path, "rb") as f_in, open(target_data_path, "rb") as f_out:
        count    = 0
        data_in  = []
        data_out = []
        len_in   = []
        len_out  = []
        for lin, lout in zip(f_in, f_out):
            if count == batch_size:
                yield data_in, len_in, data_out, len_out
                count    = 0
                data_in  = []
                data_out = []
                len_in   = []
                len_out  = []
            in_tokens  = read_wordpieces(vocab, lin, True)
            out_tokens = read_wordpieces(vocab, lout, False)
            # put translate tokens at end of sentence
            #in_tokens  = in_tokens[::-1]
            in_tokens  = in_tokens + [_EOS]
            out_tokens = [_BOS] + out_tokens + [_EOS]
            if len(in_tokens) <= max_length and len(out_tokens) <= max_length:
                len_in.append(len(in_tokens))
                in_tokens = post_pad(in_tokens, _PAD, max_length)
                len_out.append(len(out_tokens))
                out_tokens = post_pad(out_tokens, _PAD, max_length)
                data_in.append(in_tokens)
                data_out.append(out_tokens)
                count += 1

        if count == batch_size:
            yield data_in, len_in, data_out, len_out


def read_wordpieces(vocab, line, in_line):
    return [vocab[t] if t in vocab else _UNK for t in line.split()]
    #if not in_line:
    #    return [vocab[t] if t in vocab else _UNK for t in line.split()]
    #return [tmap[line[-6:-1]]] + [vocab[t] if t in vocab else _UNK for t in line[:-5].split()]


def sort_data_files(source_data_path, target_data_path):
    words = [len(line.split(" ")) for line in open(source_data_path, "rb").readlines()]
    indices = sorted(xrange(len(words)), key=lambda k: words[k])
    source_lines = open(source_data_path, "rb").readlines()
    target_lines = open(target_data_path, "rb").readlines()
    with open(source_data_path + ".sorted", "wb") as f:
        f.write("".join([source_lines[i] for i in indices]))
    with open(target_data_path + ".sorted", "wb") as f:
        f.write("".join([target_lines[i] for i in indices]))


def batch_shuffle(source_data_path, target_data_path, batch_size):
    source = open(source_data_path, "rb").readlines()
    target = open(target_data_path, "rb").readlines()
    source_batches = [source[i:i+batch_size] for i in xrange(0, len(source), batch_size)]
    target_batches = [target[i:i+batch_size] for i in xrange(0, len(target), batch_size)]
    indices = [i for i in xrange(len(source_batches))]
    shuffle(indices)
    with open(source_data_path + ".shuffled", "wb") as f:
        f.write("".join(["".join(line) for i in indices for line in source_batches[i]]))
    with open(target_data_path + ".shuffled", "wb") as f:
        f.write("".join(["".join(line) for i in indices for line in target_batches[i]]))


def prune_sentence_length(source_data_path, target_data_path, max_size):
    source = open(source_data_path, "rb").readlines()
    target = open(target_data_path, "rb").readlines()
    with open(source_data_path + ".pruned", "wb") as f_s, open(target_data_path + ".pruned", "wb") as f_t:
        for ls, lt in zip(source, target):
            if len(ls.split(" ")) < max_size and len(lt.split(" ")) < max_size:
                f_s.write(ls)
                f_t.write(lt)
