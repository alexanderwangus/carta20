import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pandas as pd
import os
import util
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import logistic_regression_course2vec
import pickle

TRAIN_LENGTH = 20
PREDICT_LENGTH = 20

def evaluate_model(X, y, model, output_dict=False, top_n=1):
    if top_n == 1:
        y_pred = model.predict(list(X))
    else:
        classes = model.classes_
        idx_to_class = {i: classes[i] for i in range(len(classes))}

        y_pred_probs = model.predict_proba(list(X))
        y_pred = (-y_pred_probs).argsort(axis=-1)[:, :top_n]

        y_pred = [[idx_to_class[idx] for idx in p] for p in y_pred]
        y_pred = top_n_conversion(y.values, y_pred)

    return classification_report(y, y_pred, zero_division=0, output_dict=output_dict)


def train_log_reg(X, y):
    reg = LogisticRegression().fit(X, y)
    train_score = reg.score(X, y)
    return train_score, reg


def multiclass_full():
    X_train, X_val, X_test, y_train, y_val, y_test = util.prep_dataset()
    train_score, reg = train_log_reg(X_train, y_train)
    print(train_score)  # 0.9915
    val_score = reg.score(X_val, y_val)
    print(val_score)  # 0.7969

    y_pred = reg.predict(X_val)
    print(classification_report(y_val, y_pred))


def multiclass_forecast():
    X_train, X_val, X_test, y_train, y_val, y_test = util.prep_dataset(num_classes=15)
    train_score, reg = train_log_reg(X_train, y_train)

    print(train_score)  # 0.8284
    val_score = reg.score(X_val, y_val)
    print(val_score)  # 0.4511

    y_pred = reg.predict(X_val)
    print(classification_report(y_val, y_pred))


'''
Trains on full course histories, but predicts on truncated course histories.
'''
def multiclass_forecast_full(num_classes_train=TRAIN_LENGTH, num_classes_predict=PREDICT_LENGTH, model=None, categories=False, top_n=1):
    X_train, X_val, X_test, y_train, y_val, y_test = util.prep_dataset_v3(num_classes_train=num_classes_train, num_classes_predict=num_classes_predict, vectorize=True)

    if categories:
        y_train = util.degrees_to_categories(y_train)
        y_val = util.degrees_to_categories(y_val)


    if not model:
        train_score, reg = train_log_reg(X_train, y_train)
        print(train_score)
        pickle.dump(reg, open("log_reg_full_train.pickle", 'wb'))
    else:

        reg = pickle.load(open(model, 'rb'))


    macro_f1 = evaluate_model(X_val, y_val, reg, output_dict=True, top_n=top_n)['macro avg']['f1-score']
    print(evaluate_model(X_val, y_val, reg, output_dict=False, top_n=top_n))
    return macro_f1, reg


def main():
    # multiclass_forecast_full(model="log_reg_full_train.pickle")
    multiclass_forecast_full(categories=False, top_n=3)


if __name__ == '__main__':
    main()
