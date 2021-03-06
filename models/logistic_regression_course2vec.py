import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../course_embeddings'))

import util
from gensim.models import Word2Vec
from logistic_regression_model import train_log_reg, top_n_conversion
from sklearn.metrics import classification_report
from course2vec import get_course2vec_model_path, train_course2vec, featurize_student, create_training_set
import numpy as np

TRAIN_LENGTH = 10
PREDICT_LENGTH = 10


def evaluate_model_bias_single_df(model, df, args, num_classes_predict=0, categories=False, top_n=1):
    course2vec_model, vec_size = args
    X, y = util.tokenize_df(df, num_classes_predict)

    X = featurize_student(X['course_history'], course2vec_model, vec_size)
    if categories:
        y = util.degrees_to_categories(y)

    return util.evaluate_model(X, y, model, output_dict=True, top_n=top_n)


def log_reg_course2vec(training_set=None, vec_size=150, win_size=10, min_count=2, epochs=10, num_classes_val=-1, categories=False, top_n=1):
    print(f"\nRunning course2vec with logreg with vec_size={vec_size}, win_size={win_size}, min_count={min_count}, epochs={epochs}, num_classes_val={num_classes_val}")

    # set up hyperparams, load model
    course2vec_model_path = get_course2vec_model_path(vec_size, win_size, min_count)

    if training_set:
        course2vec_model = train_course2vec(training_set, course2vec_model_path, vec_size, win_size, min_count, epochs=epochs)
    else:
        course2vec_model = Word2Vec.load(course2vec_model_path)

    # prep datasets
    _, X_val, X_test, _, y_val, y_test = util.prep_dataset_v3(num_classes_train=TRAIN_LENGTH, num_classes_predict=PREDICT_LENGTH, augmented=False)
    X_train, _, _, y_train, _, _ = util.prep_dataset_v3(num_classes_train=TRAIN_LENGTH, num_classes_predict=PREDICT_LENGTH, augmented=False)

    X_train = featurize_student(X_train['course_history'], course2vec_model, vec_size)
    X_val = featurize_student(X_val['course_history'], course2vec_model, vec_size)
    X_test = featurize_student(X_test['course_history'], course2vec_model, vec_size)

    if categories:
        y_train = util.degrees_to_categories(y_train)
        y_val = util.degrees_to_categories(y_val)
        y_test = util.degrees_to_categories(y_test)

    # train and predict using logistic regression model
    train_score, log_reg_model = train_log_reg(list(X_train), y_train)
    print(f"train_score: {train_score}")
    # y_pred = log_reg_model.predict(list(X_val))
    macro_f1 = util.evaluate_model(X_test, y_test, log_reg_model, output_dict=True, top_n=top_n)['macro avg']['f1-score']
    print(util.evaluate_model(X_test, y_test, log_reg_model, output_dict=False, top_n=top_n))

    util.evaluate_model_bias(log_reg_model, (course2vec_model, vec_size), evaluate_model_bias_single_df, num_classes_predict=PREDICT_LENGTH, categories=categories, top_n=top_n, test=True)

    return macro_f1, log_reg_model

def main():
    # training_set = create_training_set()
    log_reg_course2vec(training_set=None, vec_size=150, win_size=10, min_count=1, epochs=10, categories=True, top_n=1)

if __name__ == '__main__':
    main()
