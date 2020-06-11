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
