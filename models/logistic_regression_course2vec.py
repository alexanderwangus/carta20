import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import util
from gensim.models import Word2Vec
from models.logistic_regression_model import train_log_reg
from sklearn.metrics import classification_report
from course_embeddings.course2vec import get_course2vec_model_path, train_course2vec, featurize_student, create_training_set

TRAIN_LENGTH = 30
PREDICT_LENGTH = 30

def log_reg_course2vec(training_set=None, vec_size=150, win_size=10, min_count=2, epochs=10, num_classes_val=-1):
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

    # train and predict using logistic regression model
    train_score, log_reg_model = train_log_reg(list(X_train), y_train)
    print(f"train_score: {train_score}")
    y_pred = log_reg_model.predict(list(X_val))
    macro_f1 = classification_report(y_val, y_pred, output_dict=True, zero_division=0)['macro avg']['f1-score']
    print(classification_report(y_val, y_pred, zero_division=0))
    return macro_f1, log_reg_model

def main():
    # training_set = create_training_set()
    log_reg_course2vec(training_set=None, vec_size=150, win_size=10, min_count=1, epochs=10)

if __name__ == '__main__':
    main()
