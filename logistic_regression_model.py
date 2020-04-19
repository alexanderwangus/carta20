import pandas as pd
import os
import util
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle


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
def multiclass_forecast_full(num_classes_val=15, model=None):
    X_train, X_val, X_test, y_train, y_val, y_test = util.prep_dataset(num_classes_val=num_classes_val)
    if not model:
        train_score, reg = train_log_reg(X_train, y_train)
        print(train_score)
        pickle.dump(reg, open("log_reg_full_train.pickle", 'wb'))
    else:
        reg = pickle.load(open(model, 'rb'))

    val_score = reg.score(X_val, y_val)
    print(val_score)

    y_pred = reg.predict(X_val)
    print(y_pred)
    print(classification_report(y_val, y_pred))


def main():
    multiclass_forecast_full(num_classes_val=60, model="log_reg_full_train.pickle")


if __name__ == '__main__':
    main()
