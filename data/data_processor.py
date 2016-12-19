from __future__ import division
from __future__ import print_function

from unidecode import unidecode
from unicodedata import category

import string
import re


punc = string.punctuation


def process(f_in, f_out):
    new_lines = []
    for line in open(f_in, "rb"):
        # use "*" key for shuffling
        # paste -d '*' file1 file2 | shuf | awk -v FS="*" '{ print $1 > "out1" ; print $2 > "out2" }'
        line = "".join(unidecode(ch) if category(ch)[0] == "P" else ch for ch in line.decode("utf8"))
        line = line.replace("*", " ")
        for ch in punc:
            line = line.replace(ch, " " + ch + " ")
        line = re.sub("[\t ]+", " ", line)
        new_lines.append(line.strip() + "\n")

    with open(f_out, "wb") as f:
        f.write("".join(line.encode("utf8") for line in new_lines))


def append_token(f, token):
    new_lines = []
    for line in open(f, "rb"):
        new_lines.append(line.replace("\n", " " + token + "\n"))

    with open(f, "wb") as f:
        f.write("".join(line for line in new_lines))
