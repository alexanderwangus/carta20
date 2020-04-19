import pandas as pd
import numpy as np
import os
from req_builder import Req, Subreq
from sklearn.feature_extraction.text import CountVectorizer
import datetime

DATA_DIR = os.getcwd() + '/cs_tracks_notebook_data/'
MATRIX_FILE = DATA_DIR + 'crs-qtr.vec'
VOCAB_FILE = DATA_DIR + 'vocab_crs.txt'
COURSE_FILE = DATA_DIR + 'courselists/dept-CS.txt'
STUD_FILE = DATA_DIR + 'studlists/major-CS-BS.txt'
STRM_FILE = DATA_DIR + 'ID_strms.csv'
COURSE_MAJOR_FILE = DATA_DIR + 'initial_dataset.fthr'


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


'''
Vectorize student course histories.
Takes in a pandas series of course series and returns a matrix, where
each row is the given student's vectorized course history.

Returns count vectorizer and said matrix.
'''
def vectorize_course_history(srs, vectorizer=None):
    course_strings = srs.values.tolist()
    if not vectorizer:
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(course_strings)
    else:
        X = vectorizer.transform(course_strings)
    return vectorizer, X.toarray()


def truncate_class(class_string, num_classes):
    return ','.join(list(class_string.split(','))[:num_classes])  # todo: debug truncation getting rid of numbers in course code CS106A -> CS


def prep_dataset(num_classes=-1, num_classes_val=-1):
    df = pd.read_feather(COURSE_MAJOR_FILE, use_threads=True)

    if num_classes > 0:
        df['course_history'] = df['course_history'].apply(truncate_class, args=[num_classes])

    vectorizer, X = vectorize_course_history(df.loc[:, 'course_history'])
    y = df['DEGREE_1']
    # train: 2000 -> 2016 inclusive
    # val: 2017 -> 2018
    # test: 2019 -> 2020
    train_date_upper = datetime.date.fromisoformat('2016-12-31')
    val_date_upper = datetime.date.fromisoformat('2018-12-31')
    test_date_upper = datetime.date.fromisoformat('2020-12-31')

    train_indices = []
    val_indices = []
    test_indices = []

    for index, row in df.iterrows():
        if row['eff_dt_1']:
            if row['eff_dt_1'] <= train_date_upper:
                train_indices.append(index)
            elif row['eff_dt_1'] <= val_date_upper:
                val_indices.append(index)
            else:
                test_indices.append(index)

    X_train = X[train_indices]
    X_val = X[val_indices]
    X_test = X[test_indices]

    y_train = y[train_indices]
    y_val = y[val_indices]
    y_test = y[test_indices]

    if num_classes_val > 0:
        df['course_history'] = df['course_history'].apply(truncate_class, args=[num_classes_val])

    _, X_truncated = vectorize_course_history(df.loc[:, 'course_history'], vectorizer=vectorizer)

    X_val = X_truncated[val_indices]
    X_test = X_truncated[test_indices]

    y_val = y_val[val_indices]
    y_test = y_test[test_indices]

    return X_train, X_val, X_test, y_train, y_val, y_test
