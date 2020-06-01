import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import util
from gensim.models import Word2Vec
from models.logistic_regression_model import train_log_reg
from sklearn.metrics import classification_report
from course_embeddings.course2vec import get_course2vec_model_path, train_course2vec, featurize_student, create_training_set, get_course_vec
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import copy

def courses2vecs(course_list, course2vec_model, vec_size, max_length):
    vec = np.array([get_course_vec(course2vec_model, word, vec_size) for word in course_list])
    padding = 0 if len(vec) >= max_length else max_length - len(vec)
    vec = np.pad(vec, ((0, padding), (0, 0)), "constant", constant_values=0)[:max_length]
    return np.array(vec)


def expand(items, expansion_amounts):
    return [[items[i]]*expansion_amounts[i] for i in range(len(items))]


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


def subtokenize_features(df):
    for index, row in df.iterrows():
        subtokenized = subtokenize_feautures_row(row)
        df.loc[index, "course_history"] = subtokenized["course_history"]
        df.loc[index, "RELATIVE_TERM"] = subtokenized["RELATIVE_TERM"]
        df.loc[index, "CRSE_GRADE_INPUT"] = subtokenized["CRSE_GRADE_INPUT"]
    return df


def featurize_student(X, course2vec_model, vec_size, max_length):
    X = X.apply(courses2vecs, args=[course2vec_model, vec_size, max_length])
    return np.stack(X.values)


def featurize_student_v2(X, course2vec_params, max_length, training_sets=None):
    vec_size = course2vec_params["vec_size"]
    win_size = course2vec_params["win_size"]
    min_count = course2vec_params["min_count"]

    course_history_course2vec_model = Word2Vec.load(get_course2vec_model_path(vec_size, win_size, min_count, feature_type="course_history", subtokenized=subtokenize))
    term_course2vec_model = Word2Vec.load(get_course2vec_model_path(vec_size, win_size, min_count, feature_type="RELATIVE_TERM"))
    grade_course2vec_model = Word2Vec.load(get_course2vec_model_path(vec_size, win_size, min_count, feature_type="CRSE_GRADE_INPUT"))

    X_course_history = X["course_history"].apply(courses2vecs, args=[course_history_course2vec_model, vec_size, max_length])
    X_term = X["RELATIVE_TERM"].apply(courses2vecs, args=[term_course2vec_model, vec_size, max_length])
    X_grade = X["CRSE_GRADE_INPUT"].apply(courses2vecs, args=[grade_course2vec_model, vec_size, max_length])

    X_course_history = np.stack(X_course_history.values)
    X_term =  np.stack(X_term.values)
    X_grade =  np.stack(X_grade.values)

    # return X_course_history

    # return np.concatenate([X_course_history, X_term], axis=2)

    return np.concatenate([X_course_history, X_term, X_grade], axis=2)  # order to concat is X_course_history, X_term, X_grade

def get_seq_len(course_list, max_length):
    # course_list = course_str.split(',')
    if max_length > 0:
        return min(max_length, len(course_list))
    else:
        return len(course_list)


def get_X_lens(X, max_length):
    return [get_seq_len(seq.split(','), max_length) for _, seq in X.items()]

def get_X_lens_v2(X, max_length):
    return [get_seq_len(seq, max_length) for _, seq in X["course_history"].items()]


def train_model(model, X_train, X_train_lens, y_train, X_val, X_val_lens, y_val, epochs, batch_size, lr, verbose=True, categories=False, top_n=1):
    if torch.cuda.is_available():
        model = model.cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Training on {device}")

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    val_results = evaluate_model(X_val, X_val_lens, y_val, model, categories=categories, top_n=top_n)
    if verbose:
        print(f"== Pre-training val results (macro avg) ==")
        print(f"  Precision: {val_results['macro avg']['precision']}")
        print(f"  Recall: {val_results['macro avg']['recall']}")
        print(f"  f1-score: {val_results['macro avg']['f1-score']}\n")

    best_metric = -1
    best_model = None

    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            model.train()
            curr_batch_size = batch_size
            if i + batch_size > len(X_train):
                curr_batch_size = len(X_train) - i + 1

            optimizer.zero_grad()

            if categories:
                targets = torch.LongTensor([util.category_to_idx(course) for course in y_train[i:i+curr_batch_size]])
            else:
                targets = torch.LongTensor([util.course_to_idx(course) for course in y_train[i:i+curr_batch_size]])

            sentences = torch.FloatTensor(X_train[i:i+curr_batch_size])
            sentence_lens = torch.LongTensor(X_train_lens[i:i+curr_batch_size])
            if torch.cuda.is_available():
                sentences = sentences.cuda()
                sentence_lens = sentence_lens.cuda()
                targets = targets.cuda()
            probs = model(sentences, sentence_lens)
            loss = loss_function(probs, targets)
            loss.backward()
            optimizer.step()

        if verbose:
            train_results = evaluate_model(X_train, X_train_lens, y_train, model, categories=categories, top_n=top_n)
            print(f"== Epoch {epoch+1} train results (macro avg) ==")
            print(f"  Precision: {train_results['macro avg']['precision']}")
            print(f"  Recall: {train_results['macro avg']['recall']}")
            print(f"  f1-score: {train_results['macro avg']['f1-score']}")

        val_results = evaluate_model(X_val, X_val_lens, y_val, model, categories=categories, top_n=top_n)
        if verbose:
            print(f"== Epoch {epoch+1} val results (macro avg) ==")
            print(f"  Precision: {val_results['macro avg']['precision']}")
            print(f"  Recall: {val_results['macro avg']['recall']}")
            print(f"  f1-score: {val_results['macro avg']['f1-score']}\n")

        if val_results['macro avg']['f1-score'] > best_metric:
            if verbose:
                print(f"New best model found with f1-score {val_results['macro avg']['f1-score']} beating previous value of {best_metric}")
            best_model = copy.deepcopy(model)
            best_metric = val_results['macro avg']['f1-score']

    return best_model


def evaluate_model(X, X_lens, y, model, output_dict=True, categories=False, top_n=1):
    model.eval()
    batch_size = 32

    with torch.no_grad():
        if categories:
            y = torch.LongTensor([util.category_to_idx(course) for course in y])
        else:
            y = torch.LongTensor([util.course_to_idx(course) for course in y])
        y_pred = []
        for i in range(0, len(X), batch_size):
            curr_batch_size = batch_size
            if i + batch_size > len(X):
                curr_batch_size = len(X) - i + 1

            sentences = torch.FloatTensor(X[i:i+curr_batch_size])
            sentence_lens = torch.LongTensor(X_lens[i:i+curr_batch_size])
            if torch.cuda.is_available():
                sentences = sentences.cuda()
                sentence_lens = sentence_lens.cuda()
            y_pred += model.predict(sentences, sentence_lens, top_n=top_n)

        if top_n != 1:
            y_pred = util.top_n_conversion(y, y_pred)


        return classification_report(y, y_pred, zero_division=0, output_dict=output_dict)
