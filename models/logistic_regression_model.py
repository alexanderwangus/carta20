import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pandas as pd
import os
import util
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

TRAIN_LENGTH = 10
PREDICT_LENGTH = 10


def top_n_conversion(y, y_pred):
    y_top_n = []
    for i in range(len(y)):
        if y[i] in y_pred[i]:
            y_top_n.append(y[i])
        else:
            y_top_n.append(y_pred[i][0])
    return y_top_n


def evaluate_model_bias_single_df(model, df, vectorizer, num_classes_predict=0, categories=False, top_n=1):
    X, y = util.tokenize_df(df, num_classes_predict)
    _, X = util.vectorize_course_history(X.loc[:, 'course_history'], vectorizer=vectorizer)

    if categories:
        y = util.degrees_to_categories(y)

    return util.evaluate_model(X, y, model, output_dict=True, top_n=top_n)


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
    (X_train, X_val, X_test, y_train, y_val, y_test), vectorizer = util.prep_dataset_v3(num_classes_train=num_classes_train, num_classes_predict=num_classes_predict, vectorize=True)

    if categories:
        y_train = util.degrees_to_categories(y_train)
        y_val = util.degrees_to_categories(y_val)
        y_test = util.degrees_to_categories(y_test)


    if not model:
        train_score, reg = train_log_reg(X_train, y_train)
        print(train_score)
        if categories:
            pickle.dump(reg, open("log_reg_full_train_categories.pickle", 'wb'))
        else:
            pickle.dump(reg, open("log_reg_full_train.pickle", 'wb'))
    else:

        reg = pickle.load(open(model, 'rb'))

    macro_f1 = util.evaluate_model(X_test, y_test, reg, output_dict=True, top_n=top_n)['macro avg']['f1-score']
    print(util.evaluate_model(X_test, y_test, reg, output_dict=False, top_n=top_n))

    util.evaluate_model_bias(reg, vectorizer, evaluate_model_bias_single_df, num_classes_predict=PREDICT_LENGTH, categories=False, top_n=top_n, test=True)

    return macro_f1, reg


def main():
    # multiclass_forecast_full(model="log_reg_full_train_categories.pickle", top_n=1)
    multiclass_forecast_full(categories=True, top_n=1)


if __name__ == '__main__':
    main()
