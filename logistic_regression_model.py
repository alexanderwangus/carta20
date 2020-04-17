import pandas as pd
import os
import util
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def train_log_reg(X, y):
    reg = LogisticRegression().fit(X, y)
    train_score = reg.score(X, y)
    return train_score, reg


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = util.prep_dataset()
    train_score, reg = train_log_reg(X_train, y_train)
    print(train_score)  # 0.9915
    val_score = reg.score(X_val, y_val)
    print(val_score)  # 0.7969

    y_pred = reg.predict(X_val)
    print(classification_report(y_val, y_pred))


if __name__ == '__main__':
    main()

"""
              precision    recall  f1-score   support

          AA       0.00      0.00      0.00         6
       AFRAM       0.55      0.25      0.34        24
       AMSTU       0.88      0.81      0.85        27
       ANTHR       0.91      0.72      0.81        29
       ARCHA       1.00      0.75      0.86         4
         ART       0.00      0.00      0.00         0
       ARTHS       0.94      0.70      0.80        23
        ARTP       0.71      0.41      0.52        29
        ASAM       0.00      0.00      0.00         1
         BIO       0.95      0.76      0.84       140
        BIOE       1.00      0.02      0.03        59
        BIOL       0.00      0.00      0.00         0
          CE       0.85      0.88      0.86        32
        CHEM       0.85      0.87      0.86        39
       CHEME       0.90      0.95      0.92        73
       CHILT       0.00      0.00      0.00         2
       CHINE       0.00      0.00      0.00         2
       CLASS       0.82      0.43      0.56        21
       COMMU       0.82      0.91      0.86        54
       CPLIT       0.78      0.82      0.80        17
      CRWRIT       0.00      0.00      0.00         2
          CS       0.86      0.90      0.88       823
        CSRE       0.28      0.31      0.29        16
       EASST       0.00      0.00      0.00         9
       EASYS       0.75      0.92      0.83        90
        ECON       0.82      0.89      0.85       156
          ED       0.00      0.00      0.00         0
          EE       0.83      0.88      0.86       138
        ENGL       0.75      0.80      0.78       100
        ENGR       0.64      0.86      0.73       285
       ENVSE       0.00      0.00      0.00        21
         ERE       0.67      0.80      0.73         5
        FGSS       1.00      0.10      0.18        10
        FILM       0.62      0.50      0.55        16
       FRENC       0.33      0.38      0.35         8
       GEOPH       1.00      0.60      0.75         5
       GERST       0.00      0.00      0.00         1
         GES       0.80      0.67      0.73         6
      GLBLST       0.00      0.00      0.00         2
          GS       0.00      0.00      0.00         3
       HSTRY       0.80      0.84      0.82        75
       HUMBI       0.83      0.97      0.89       261
       IDMEN       0.00      0.00      0.00        10
          IE       0.00      0.00      0.00         0
        ILAC       1.00      0.25      0.40         4
       INSST       0.00      0.00      0.00         1
       INTLR       0.75      0.77      0.76       114
        ITAL       0.00      0.00      0.00         3
       JAPAN       0.00      0.00      0.00         1
        LING       0.78      0.50      0.61        14
       MATCS       0.74      0.77      0.76       102
        MATH       0.69      0.59      0.64        64
       MATSC       1.00      0.79      0.88        19
          ME       0.92      0.85      0.89       185
       MGTSC       0.90      0.78      0.83       178
      MODLAN       0.00      0.00      0.00         2
       MUSIC       0.50      0.72      0.59        18
       NATAM       1.00      0.50      0.67         2
       PHILO       0.73      0.73      0.73        30
       PHREL       1.00      0.29      0.44         7
        PHYS       0.77      0.87      0.82        75
       POLSC       0.79      0.69      0.73       118
       PSYCH       0.78      0.85      0.81       121
       PUBPO       0.79      0.71      0.75        58
       RELST       0.00      0.00      0.00         2
        SLAV       0.50      0.20      0.29         5
       SOCIO       0.69      0.50      0.58        18
        SPAN       0.00      0.00      0.00         2
         STS       0.78      0.82      0.80       165
       SYMBO       0.76      0.76      0.76       174
       THPST       0.77      0.59      0.67        17
       URBST       0.77      0.77      0.77        22

    accuracy                           0.80      4145
   macro avg       0.56      0.47      0.49      4145
weighted avg       0.80      0.80      0.78      4145
"""
