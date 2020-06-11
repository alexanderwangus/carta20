import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import util
import deep_model_util
from gensim.models import Word2Vec
from models.logistic_regression_model import train_log_reg
from sklearn.metrics import classification_report
from course_embeddings.course2vec import get_course2vec_model_path, train_course2vec, featurize_student, create_training_set, get_course_vec
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import copy

"""
This file defines helper fns for data processing and prep to use with models
that take in course2vec features. All fns to do with actually training a
course2vec model should be in course2vec.py
"""


def courses2vecs(course_list, course2vec_model, vec_size, max_length):
    vec = np.array([get_course_vec(course2vec_model, word, vec_size) for word in course_list])
    padding = 0 if len(vec) >= max_length else max_length - len(vec)
    vec = np.pad(vec, ((0, padding), (0, 0)), "constant", constant_values=0)[:max_length]
    return np.array(vec)

# TODO: move to util.py
def expand(items, expansion_amounts):
    return [[items[i]]*expansion_amounts[i] for i in range(len(items))]

# TODO: move to util.py
def subtokenize_feautures_row(row):
    courses = row["course_history"]
    terms = row["RELATIVE_TERM"]
    grades = row["CRSE_GRADE_INPUT"]

    # note: can speed up by returning list of lists of sizes, and use that to vector op expansions of terms and grades
    subtokenized_courses = [util.subtokenize_single_course(course) for course in courses]
    expansion_amounts = [len(l) for l in subtokenized_courses]
    expanded_terms = expand(terms, expansion_amounts)
    expanded_grades = expand(grades, expansion_amounts)

    row["course_history"] = [item for sublist in subtokenized_courses for item in sublist]
    row["RELATIVE_TERM"] = [item for sublist in expanded_terms for item in sublist]
    row["CRSE_GRADE_INPUT"] = [item for sublist in expanded_grades for item in sublist]

    return row

# TODO: move to util.py
def subtokenize_features(df):
    for index, row in df.iterrows():
        subtokenized = subtokenize_feautures_row(row)
        df.loc[index, "course_history"] = subtokenized["course_history"]
        df.loc[index, "RELATIVE_TERM"] = subtokenized["RELATIVE_TERM"]
        df.loc[index, "CRSE_GRADE_INPUT"] = subtokenized["CRSE_GRADE_INPUT"]
    return df

# TODO: rename to be more descriptive of course2vec featurization
def featurize_student(X, course2vec_model, vec_size, max_length):
    X = X.apply(courses2vecs, args=[course2vec_model, vec_size, max_length])
    return np.stack(X.values)

# TODO: rename to be more descriptive of course2vec featurization
def featurize_student_v2(X, course2vec_params, max_length, training_sets=None, subtokenize=False):
    vec_size = course2vec_params["vec_size"]
    win_size = course2vec_params["win_size"]
    min_count = course2vec_params["min_count"]

    course_history_course2vec_model = Word2Vec.load(get_course2vec_model_path(vec_size, win_size, min_count, feature_type="course_history", subtokenized=subtokenize))
    term_course2vec_model = Word2Vec.load(get_course2vec_model_path(vec_size, win_size, min_count, feature_type="RELATIVE_TERM"))
    grade_course2vec_model = Word2Vec.load(get_course2vec_model_path(vec_size, win_size, min_count, feature_type="CRSE_GRADE_INPUT"))

    if subtokenize:
        X = subtokenize_features(X)
        max_length = max_length * 4
    X_course_history = X["course_history"].apply(courses2vecs, args=[course_history_course2vec_model, vec_size, max_length])
    X_term = X["RELATIVE_TERM"].apply(courses2vecs, args=[term_course2vec_model, vec_size, max_length])
    X_grade = X["CRSE_GRADE_INPUT"].apply(courses2vecs, args=[grade_course2vec_model, vec_size, max_length])

    X_course_history = np.stack(X_course_history.values)
    X_term =  np.stack(X_term.values)
    X_grade =  np.stack(X_grade.values)

    # return X_course_history

    # return np.concatenate([X_course_history, X_term], axis=2)

    return np.concatenate([X_course_history, X_term, X_grade], axis=2)  # order to concat is X_course_history, X_term, X_grade
