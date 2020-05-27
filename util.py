import pandas as pd
import numpy as np
import os
from req_builder import Req, Subreq
from sklearn.feature_extraction.text import CountVectorizer
import datetime
import re

DATA_DIR = os.getcwd() + '/data/'
MATRIX_FILE = DATA_DIR + 'crs-qtr.vec'
VOCAB_FILE = DATA_DIR + 'vocab_crs.txt'
COURSE_FILE = DATA_DIR + 'courselists/dept-CS.txt'
STUD_FILE = DATA_DIR + 'studlists/major-CS-BS.txt'
STRM_FILE = DATA_DIR + 'ID_strms.csv'
COURSE_MAJOR_FILE = DATA_DIR + 'initial_dataset.fthr'
RAW_DATA_FILE = DATA_DIR + 'course_outcomes.tsv'
COURSE_OUTCOME_LIST_FILE = DATA_DIR + 'course_outcome_lists.pkl'
COURSE_OUTCOME_LIST_FILE_AUGMENTED = DATA_DIR + 'course_outcome_lists_augmented_2.pkl'


MAJOR_LIST = ['BIOE', 'FILM', 'POLSC', 'CEE', 'HUMBI', 'CS', 'MATH', 'LAMER', 'EASST', 'ANSCI', 'AMSTU', 'MODLAN', 'PHYS', 'COMMU', 'ENVSE', 'INTLR', 'HUMAN', 'ASAM', 'DRAMA', 'CLASS', 'VTSS', 'IDMJR', 'PORT', 'ARTHS', 'SOCIS', 'ECON', 'IE', 'GS', 'GEOPH', 'ENVEN', 'IDMHS', 'HSTRY', 'FRENC', 'HUMRTS', 'MATCS', 'CE', 'ERE', 'GLBLST', 'POLSS', 'ENGR', 'ENGLI', 'COMMUS', 'CRWRIT', 'CHEM', 'LING', 'CHICA', 'INSST', 'PUBPO', 'PSYCH', 'FEMST', 'ARCHA', 'AFRAM', 'ETHSO', 'SOCIO', 'AA', 'NATAM', 'MATSC', 'ITAL', 'PHREL', 'PHILO', 'SPAN', 'ENGLF', 'STS', 'URBST', 'EASYS', 'CASA', 'AFRST', 'ANTHS', 'ENGLG', 'JAPAN', 'ENGL', 'MGTSC', 'BIOL', 'PETEN', 'CHILT', 'ANTHR', 'MELLC', 'ART', 'ME', 'CHINE', 'EE', 'FRENI', 'EDUC', 'ARTP', 'RELST', 'BIO', 'ILAC', 'ED', 'MUSIC', 'GERST', 'CSRE', 'FGSS', 'CPLIT', 'CHEME', 'HUMLG', 'SLAV', 'THPST', 'IDSH', 'SYMBO', 'ESTP', 'IDMEN', 'GES', 'AMELLC', 'ENGLS']

NUM_CLASSES = len(MAJOR_LIST) + 1

COURSE_TO_IDX = {MAJOR_LIST[i]: i+1 for i in range(len(MAJOR_LIST))}
IDX_TO_COURSE = {i+1: MAJOR_LIST[i] for i in range(len(MAJOR_LIST))}

def course_to_idx(course):
    if course in COURSE_TO_IDX:
        return COURSE_TO_IDX[course]
    else:
        return 0

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
    return ','.join(list(class_string.split(','))[:num_classes])


def truncate_class_v2(class_list, num_classes):
    return class_list[:num_classes]


def get_vocab():
    df = pd.read_feather(COURSE_MAJOR_FILE, use_threads=True)
    print(df['DEGREE_1'].values)
    return set(df['DEGREE_1'].values)


def train_test_split(X, y, df):
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

    X_train = X.loc[train_indices]  # NOTE: probably broken if called from v1, since series have no loc fn
    X_val = X.loc[val_indices]
    X_test = X.loc[test_indices]

    y_train = y[train_indices]
    y_val = y[val_indices]
    y_test = y[test_indices]

    return (X_train, X_val, X_test, y_train, y_val, y_test), (train_indices, val_indices, test_indices)


def prep_dataset_v2(num_classes_train=-1, num_classes_predict=-1, augmented=False):
    if augmented:
        df = pd.read_pickle(COURSE_OUTCOME_LIST_FILE_AUGMENTED)
    else:
        df = pd.read_pickle(COURSE_OUTCOME_LIST_FILE)
    df['eff_dt_1'] = df['eff_dt_1'].apply(datetime.date.fromisoformat)

    # print(df.columns)

    y = df['ACAD_PLAN_1']
    X = df.loc[:, ['course_history', 'RELATIVE_TERM', 'CRSE_GRADE_INPUT']]

    (X_train, X_val, X_test, y_train, y_val, y_test), (train_indices, val_indices, test_indices) = train_test_split(X, y, df)

    if num_classes_predict > 0 or num_classes_train > 0:
        df['course_history'] = df['course_history'].apply(truncate_class_v2, args=[num_classes_predict])
        df['RELATIVE_TERM'] = df['RELATIVE_TERM'].apply(truncate_class_v2, args=[num_classes_predict])
        df['CRSE_GRADE_INPUT'] = df['CRSE_GRADE_INPUT'].apply(truncate_class_v2, args=[num_classes_predict])
        X_truncated = df.loc[:, ['course_history', 'RELATIVE_TERM', 'CRSE_GRADE_INPUT']]

        if num_classes_train > 0:
            X_train = X_truncated.loc[train_indices]
            # y_train = y_train[train_indices]

        if num_classes_predict > 0:
            X_val = X_truncated.loc[val_indices]
            X_test = X_truncated.loc[test_indices]
            # y_val = y_val[val_indices]
            # y_test = y_test[test_indices]

    return X_train, X_val, X_test, y_train, y_val, y_test


def prep_dataset(num_classes_predict=-1, vectorize=False):
    df = pd.read_feather(COURSE_MAJOR_FILE, use_threads=True)

    if vectorize:
        vectorizer, X = vectorize_course_history(df.loc[:, 'course_history'])
    else:
        X = df.loc[:, 'course_history']
    y = df['DEGREE_1']

    (X_train, X_val, X_test, y_train, y_val, y_test), (train_indices, val_indices, test_indices) = train_test_split(X, y, df)

    if num_classes_predict > 0:
        df['course_history'] = df['course_history'].apply(truncate_class, args=[num_classes_predict])

    if vectorize:
        _, X_truncated = vectorize_course_history(df.loc[:, 'course_history'], vectorizer=vectorizer)
    else:
        X_truncated = df.loc[:, 'course_history']

    X_val = X_truncated[val_indices]
    X_test = X_truncated[test_indices]

    y_val = y_val[val_indices]
    y_test = y_test[test_indices]

    return X_train, X_val, X_test, y_train, y_val, y_test


# returns list of subtokens
def subtokenize_single_course(course_str):
    match = re.match(r"([^0-9]+)([0-9]+)([^0-9]*)", course_str, re.I)
    # print(course_str)
    items = list(match.groups())
    if items[-1] == "":
        items = items[:-1]
    if len(items[1]) == 3:
        items.insert(1, items[1][0])
        items[2] = items[2][1:]
    return items


# returns list of subtokens
def subtokenize_single_course_v2(course_str):
    match = re.match(r"([^0-9]+)([0-9]+)([^0-9]*)", course_str, re.I)
    items = list(match.groups())
    items[1] = items[1] + items[2]

    if items[-1] == "":
        items = items[:-1]

    print(items)

    return items


def main():
    print(get_vocab())

if __name__ == '__main__':
    main()
