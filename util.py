import pandas as pd
import numpy as np
import os
from req_builder import Req, Subreq

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


'''
Filter out students from dataframe X that do not fulfill given requirement req
'''
def filter_req(X, req, rev_vocab):
    if isinstance(req, tuple):
        print(req[0].reqs)
    if req.all:
        for r in req.reqs:
            if isinstance(r, str):
                if r not in rev_vocab:
                    print(r, 'not in rev_vocab')
                    continue
                courseix = rev_vocab[r]
                X = X[X.iloc[:,courseix] > 0]
            elif isinstance(r, (Req, Subreq)):
                stud_idx = filter_req(X, r, rev_vocab)
                X = X[X.index.isin(stud_idx)]
            else:
                print(type(r), r.reqs)
                raise TypeError
        return X.index
    else:
        stud_filters = []
        for r in req.reqs:
            if isinstance(r, str):
                if r not in rev_vocab:
                    print(r, 'not in rev_vocab')
                    continue
                courseix = rev_vocab[r]
                stud_filters.append(X.iloc[:,courseix] > 0)
            elif isinstance(r, (Req, Subreq)):
                stud_idx = filter_req(X, r, rev_vocab)
                stud_filters.append(X.index.isin(stud_idx))
            else:
                print(type(r), r.reqs)
                raise TypeError
        filter_df = pd.DataFrame(stud_filters).T
        return X.index[filter_df.sum(axis=1) >= req.num]


'''
Wrapper for filter_req
'''
def get_studlist(X, req, filename, rev_vocab):
    stud_idx = filter_req(X, req, rev_vocab)
    matches = stud_idx.to_series()
    return matches


'''
Assigns students uniquely to different tracks
'''
def assign_unique_students(tracknames, track_students, track_priorities):
    track_students_index = {}
    for t in tracknames:
        track_students_index[t] = pd.Index(track_students[t].values)

    track_students_unique = {}
    for t1 in tracknames:
        idx1 = track_students_index[t1]
        for t2 in tracknames:
            if t1 != t2:
                idx1 = idx1.difference(track_students_index[t2])
                if track_priorities[t1] < track_priorities[t2]:
                    overlap = track_students_index[t1].intersection(track_students_index[t2])
                    idx1 = idx1.union(overlap)
        track_students_unique[t1] = idx1
    return track_students_unique

'''
Count number of courses that deviate from each req (i.e. no. of courses needed to fulfill req)
'''
def get_req_deviation(stud_row, req, rev_vocab):
    dev = 0
    for r in req.reqs:
        if isinstance(r, str):
            if r not in rev_vocab:
                continue
            courseix = rev_vocab[r]
            if stud_row[courseix] <=0:
                dev += 1
        elif isinstance(r, Req) or isinstance(r, Subreq):
            dev += get_req_deviation(stud_row, r, rev_vocab)
        else:
            print(type(r), r.reqs)
            raise TypeError
    if not req.all:
        dev -= (len(req.reqs) - req.num) # TODO: debug. if a course is present in several reqs this algorithm looks like it double counts it
    return max(0, dev)


'''
Count number of courses that deviate from each track (i.e. no. of courses needed to fulfill track)
'''
def get_tracks_deviation(stud_row, tracknames, tracks, rev_vocab):
    tracks_dev = []
    for t in tracknames:
        tracks_dev.append(get_req_deviation(stud_row, tracks[t], rev_vocab))
    return pd.Series(tracks_dev)
