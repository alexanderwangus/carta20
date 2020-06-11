import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pandas as pd
import os
import util
from sklearn.dummy import DummyClassifier
import logistic_regression_model

TRAIN_LENGTH = 10
PREDICT_LENGTH = 10

'''
Stratified guesser baseline
'''
def run_baseline_classifier(num_classes_train=TRAIN_LENGTH, num_classes_predict=PREDICT_LENGTH, categories=False, top_n=1):
    (X_train, X_val, X_test, y_train, y_val, y_test), vectorizer = util.prep_dataset_v3(num_classes_train=num_classes_train, num_classes_predict=num_classes_predict, vectorize=True)

    if categories:
        y_train = util.degrees_to_categories(y_train)
        y_val = util.degrees_to_categories(y_val)
        y_test = util.degrees_to_categories(y_test)


    dummy_clf = DummyClassifier(strategy='stratified').fit(X_train, y_train)

    macro_f1 = util.evaluate_model(X_test, y_test, dummy_clf, output_dict=True, top_n=top_n)['macro avg']['f1-score']
    print(util.evaluate_model(X_test, y_test, dummy_clf, output_dict=False, top_n=top_n))

    util.evaluate_model_bias(dummy_clf, vectorizer, logistic_regression_model.evaluate_model_bias_single_df, num_classes_predict=PREDICT_LENGTH, categories=categories, top_n=top_n, test=True)

    return macro_f1, dummy_clf


def main():
    run_baseline_classifier(categories=True, top_n=1)


if __name__ == '__main__':
    main()
