import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import util
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import models.logistic_regression_model
from sklearn.metrics import classification_report
from ast import literal_eval

"""
This file defines all functionality needed to train a course2vec model.
Util fns related to models downstream of this should be in course2vec_util.py
"""


def to_strlist(l):
    return [str(x) for x in l]


def subtokenize_course_history(course_history):
    subtokenized = [util.subtokenize_single_course(s) for s in course_history]
    return [item for sublist in subtokenized for item in sublist]


"""
Features can be 'course_history', 'RELATIVE_TERM', or 'CRSE_GRADE_INPUT'
"""
def create_training_set(dataset=util.COURSE_OUTCOME_LIST_FILE, feature_type='course_history', truncation=-1):
    # df = pd.read_feather(dataset, use_threads=True)
    df = pd.read_pickle(dataset)
    if truncation > 0:
        df = df[:truncation]

    df[feature_type] = df[feature_type].apply(to_strlist)
    df[feature_type] = df[feature_type].apply(subtokenize_course_history)

    # df[feature_type] = df[feature_type].apply(literal_eval)
    # training_set = [sentence.split(',') for sentence in list(df[feature_type])]
    training_set = list(df[feature_type])
    print(training_set)

    return training_set


def create_model(training_set, vec_size=150, win_size=10, min_count=2):
    model = Word2Vec(
        training_set,
        size=vec_size,
        window=win_size,
        min_count=min_count,
        workers=10,
        batch_words=5000)
    return model


def train_course2vec(training_set, model_path, vec_size, win_size, min_count, epochs=10):
    model = create_model(training_set, vec_size, win_size, min_count)
    model.train(training_set, total_examples=len(training_set), epochs=epochs)
    model.save(model_path)
    return model


"""
feature_type can be 'course_history', 'RELATIVE_TERM', or 'CRSE_GRADE_INPUT'
"""
def get_course2vec_model_path(vec_size, win_size, min_count, epochs=10, feature_type="course_history", subtokenized=False):
    if subtokenized:
        return f"course2vec_saved_models/subtokenized/{feature_type}_word2vec_vec{vec_size}_win{win_size}_min{min_count}.model"
    else:
        return f"course2vec_saved_models/{feature_type}_word2vec_vec{vec_size}_win{win_size}_min{min_count}.model"


def get_course_vec(model, word, vec_size):
    return model.wv[word] if word in model.wv else np.zeros(vec_size)


def courses2vec(course_list, model, vec_size):
    vec = np.mean([get_course_vec(model, word, vec_size) for word in course_list], axis=0)
    return np.array(vec)


def featurize_student(X, model, vec_size, truncation=-1):
    X = X.apply(courses2vec, args=[model, vec_size])
    return X


def hyperparam_search(feature_type):
    training_set = create_training_set(feature_type=feature_type)

    vec_sizes = [100, 150, 200, 300]
    win_sizes = [2, 5, 10]
    min_counts = [1, 2, 5, 10]

    best_metric = -1
    best_config = {}
    best_model = None

    for vec_size in vec_sizes:
        for win_size in win_sizes:
            for min_count in min_counts:
                print(f"Running trial with vec_size: {vec_size}, win_size: {win_size}, min_count: {min_count}")
                metric, model = log_reg_course2vec(training_set, vec_size, win_size, min_count)
                if metric > best_metric:
                    print(f"New best metric of {metric} to beat old metric of {best_metric} found.")
                    best_metric = metric
                    best_config = {"vec_size": vec_size, "win_size": win_size, "min_count": min_count}
                    print(f"New best config: {best_config}")
                    best_model = model
                print("\n")

    model.save("word2vec_best.model")

def main():
    vec_size=150
    win_size=10
    min_count=1
    epochs=10
    feature_type = 'course_history'

    course2vec_model_path = get_course2vec_model_path(vec_size, win_size, min_count, epochs=10, feature_type=feature_type)
    training_set = create_training_set(feature_type=feature_type)
    course2vec_model = train_course2vec(training_set, course2vec_model_path, vec_size, win_size, min_count, epochs=epochs)

if __name__ == '__main__':
    main()
