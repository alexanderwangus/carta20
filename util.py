import pandas as pd
import numpy as np
import os

DATA_DIR = os.getcwd() + '/cs_tracks_notebook_data/'
MATRIX_FILE = DATA_DIR + 'crs-qtr.vec'
VOCAB_FILE = DATA_DIR + 'vocab_crs.txt'
COURSE_FILE = DATA_DIR + 'courselists/dept-CS.txt'
STUD_FILE = DATA_DIR + 'studlists/major-CS-BS.txt'
STRM_FILE = DATA_DIR + 'ID_strms.csv'


def load_vocab(path):
    with open(path) as f:
        vocab = f.readlines()
        vocab = [w.strip().upper() for w in vocab]
    return vocab

def load_rev_vocab(vocab):
    rev_vocab = dict([(x, y) for (y, x) in enumerate(vocab)])
    return rev_vocab

def load_courselist(path, limit=None):
    with open(path) as f:
        vocab = f.readlines()
        vocab = [w.strip().upper() for w in vocab]
    if limit:
        return vocab[:limit]
    return vocab
