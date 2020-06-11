import pandas as pd
import numpy as np
import os
from req_builder import Req, Subreq
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
import datetime
import re
import torch

DATA_DIR = os.getcwd() + '/data/'
BIAS_DIR = os.getcwd() + '/data/bias_testing/'

# Data v1
COURSE_MAJOR_FILE = DATA_DIR + 'initial_dataset.fthr'
RAW_DATA_FILE = DATA_DIR + 'course_outcomes.tsv'

# Data v2
COURSE_OUTCOME_LIST_FILE = DATA_DIR + 'course_outcome_lists.pkl'
COURSE_OUTCOME_LIST_FILE_AUGMENTED = DATA_DIR + 'course_outcome_lists_augmented_2.pkl'

# Data v2, but predetermined random splits
COURSE_OUTCOME_LIST_TRAIN_FILE = DATA_DIR + 'course_outcome_lists_train.pkl'
COURSE_OUTCOME_LIST_VAL_FILE = DATA_DIR + 'course_outcome_lists_val.pkl'
COURSE_OUTCOME_LIST_TEST_FILE = DATA_DIR + 'course_outcome_lists_test.pkl'

DEGREE_CATEGORY_FILE = DATA_DIR + 'degree_to_degree_category.csv'

GENDER_STEM_STEREOTYPE_VAL_FILE = BIAS_DIR + 'gender_stem_stereotype_val.pkl'
GENDER_STEM_ANTI_STEREOTYPE_VAL_FILE = BIAS_DIR + 'gender_stem_anti_stereotype_val.pkl'
GENDER_STEM_STEREOTYPE_TEST_FILE = BIAS_DIR + 'gender_stem_stereotype_test.pkl'
GENDER_STEM_ANTI_STEREOTYPE_TEST_FILE = BIAS_DIR + 'gender_stem_anti_stereotype_test.pkl'
MALE_VAL_FILE = BIAS_DIR + 'male_val.pkl'
FEMALE_VAL_FILE = BIAS_DIR + 'female_val.pkl'
HIGH_GPA_VAL_FILE = BIAS_DIR + 'high_gpa_val.pkl'
LOW_GPA_VAL_FILE = BIAS_DIR + 'low_gpa_val.pkl'

GPA_STEM_STEREOTYPE_VAL_FILE = BIAS_DIR + 'gpa_stem_stereotype_val.pkl'
GPA_STEM_ANTI_STEREOTYPE_VAL_FILE = BIAS_DIR + 'gpa_stem_anti_stereotype_val.pkl'
GPA_STEM_STEREOTYPE_TEST_FILE = BIAS_DIR + 'gpa_stem_stereotype_test.pkl'
GPA_STEM_ANTI_STEREOTYPE_TEST_FILE = BIAS_DIR + 'gpa_stem_anti_stereotype_test.pkl'
MALE_TEST_FILE = BIAS_DIR + 'male_test.pkl'
FEMALE_TEST_FILE = BIAS_DIR + 'female_test.pkl'
HIGH_GPA_TEST_FILE = BIAS_DIR + 'high_gpa_test.pkl'
LOW_GPA_TEST_FILE = BIAS_DIR + 'low_gpa_test.pkl'


MAJOR_LIST = ['BIOE', 'FILM', 'POLSC', 'CEE', 'HUMBI', 'CS', 'MATH', 'LAMER', 'EASST', 'ANSCI', 'AMSTU', 'MODLAN', 'PHYS', 'COMMU', 'ENVSE', 'INTLR', 'HUMAN', 'ASAM', 'DRAMA', 'CLASS', 'VTSS', 'IDMJR', 'PORT', 'ARTHS', 'SOCIS', 'ECON', 'IE', 'GS', 'GEOPH', 'ENVEN', 'IDMHS', 'HSTRY', 'FRENC', 'HUMRTS', 'MATCS', 'CE', 'ERE', 'GLBLST', 'POLSS', 'ENGR', 'ENGLI', 'COMMUS', 'CRWRIT', 'CHEM', 'LING', 'CHICA', 'INSST', 'PUBPO', 'PSYCH', 'FEMST', 'ARCHA', 'AFRAM', 'ETHSO', 'SOCIO', 'AA', 'NATAM', 'MATSC', 'ITAL', 'PHREL', 'PHILO', 'SPAN', 'ENGLF', 'STS', 'URBST', 'EASYS', 'CASA', 'AFRST', 'ANTHS', 'ENGLG', 'JAPAN', 'ENGL', 'MGTSC', 'BIOL', 'PETEN', 'CHILT', 'ANTHR', 'MELLC', 'ART', 'ME', 'CHINE', 'EE', 'FRENI', 'EDUC', 'ARTP', 'RELST', 'BIO', 'ILAC', 'ED', 'MUSIC', 'GERST', 'CSRE', 'FGSS', 'CPLIT', 'CHEME', 'HUMLG', 'SLAV', 'THPST', 'IDSH', 'SYMBO', 'ESTP', 'IDMEN', 'GES', 'AMELLC', 'ENGLS']

CATEGORY_LIST_OLD = ['EARTHSCI', 'EDUCATION', 'ENGR', 'H&S', 'H&S-HUM&ART', 'H&S-INTERDISC', 'H&S-NATSCI', 'H&S-SOCSCI', 'H&S-RESEARCH', 'INDIVIDUAL', 'INTERDISC', 'MEDICINE', 'UNDECLARED', 'OTHER']

CATEGORY_LIST = ['EARTHSCI', 'ENGR', 'H&S-HUM&ART', 'H&S-INTERDISC', 'H&S-NATSCI', 'H&S-SOCSCI', 'OTHER']

NUM_CLASSES = len(MAJOR_LIST) + 1
NUM_CATEGORIES = len(CATEGORY_LIST) + 1

COURSE_TO_IDX = {MAJOR_LIST[i]: i+1 for i in range(len(MAJOR_LIST))}
IDX_TO_COURSE = {i+1: MAJOR_LIST[i] for i in range(len(MAJOR_LIST))}

CATEGORY_TO_IDX = {CATEGORY_LIST[i]: i+1 for i in range(len(CATEGORY_LIST))}
IDX_TO_CATEGORY = {i+1: CATEGORY_LIST[i] for i in range(len(CATEGORY_LIST))}


def course_to_idx(course):
    if course in COURSE_TO_IDX:
        return COURSE_TO_IDX[course]
    else:
        return 0


def category_to_idx(category):
    if category in CATEGORY_TO_IDX:
        return CATEGORY_TO_IDX[category]
    else:
        return 0


'''
Vectorize student course histories (each element is lists of strs).
Takes in a pandas series of course series and returns a matrix, where
each row is the given student's vectorized course history.

Returns count vectorizer and said matrix.
'''
def vectorize_course_history(srs, vectorizer=None):
    srs = srs.apply(lambda x: ' '.join(x))
    course_strings = srs.values.tolist()
    if not vectorizer:
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(course_strings)
    else:
        X = vectorizer.transform(course_strings)
    return vectorizer, X.toarray()


"""
DEPRECATED
Used on old dataset to word tokenize
"""
def truncate_class_v1(class_string, num_classes):
    return ','.join(list(class_string.split(','))[:num_classes])

def truncate_class(class_list, num_classes):
    return class_list[:num_classes]


def get_seq_len(course_list, max_length):
    if max_length > 0:
        return min(max_length, len(course_list))
    else:
        return len(course_list)


def get_X_lens(X, max_length):
    return [get_seq_len(seq, max_length) for _, seq in X["course_history"].items()]


def get_vocab():
    df = pd.read_feather(COURSE_MAJOR_FILE, use_threads=True)
    print(df['DEGREE_1'].values)
    return set(df['DEGREE_1'].values)


"""
Word tokenizes dataframes, and truncates sequences if specified
"""
def tokenize_df(df, num_classes):
    y = df['ACAD_PLAN_1']

    X = df.loc[:, ['course_history', 'RELATIVE_TERM', 'CRSE_GRADE_INPUT']]

    X['course_history'] = X['course_history'].apply(word_tokenize)
    X['RELATIVE_TERM'] = X['RELATIVE_TERM'].apply(word_tokenize)
    X['CRSE_GRADE_INPUT'] = X['CRSE_GRADE_INPUT'].apply(word_tokenize)

    if num_classes > 0:
        X['course_history'] = X['course_history'].apply(truncate_class, args=[num_classes])

    return X, y

"""
Prepares datasets (same as v2) that are randomly split.
"""
def prep_dataset_v3(num_classes_train=-1, num_classes_predict=-1, augmented=False, vectorize=False):
    df_train = pd.read_pickle(COURSE_OUTCOME_LIST_TRAIN_FILE)
    df_val = pd.read_pickle(COURSE_OUTCOME_LIST_VAL_FILE)
    df_test = pd.read_pickle(COURSE_OUTCOME_LIST_TEST_FILE)

    X_train, y_train = tokenize_df(df_train, num_classes_train)
    X_val, y_val = tokenize_df(df_val, num_classes_predict)
    X_test, y_test = tokenize_df(df_test, num_classes_predict)

    if vectorize:
        vectorizer, X_train = vectorize_course_history(X_train.loc[:, 'course_history'])
        _, X_val = vectorize_course_history(X_val.loc[:, 'course_history'], vectorizer=vectorizer)
        _, X_test = vectorize_course_history(X_test.loc[:, 'course_history'], vectorizer=vectorizer)
        return (X_train, X_val, X_test, y_train, y_val, y_test), vectorizer

    return X_train, X_val, X_test, y_train, y_val, y_test


"""
Helper fn for prep_dataset_v2 that carries out the date splitting
"""
def date_split(X, y, df):
    train_date_upper = datetime.date.fromisoformat('2016-12-31')  # train: 2000 -> 2016 inclusive
    val_date_upper = datetime.date.fromisoformat('2018-12-31')  # val: 2017 -> 2018 inclusive
    test_date_upper = datetime.date.fromisoformat('2020-12-31')  # test: 2019 -> 2020 inclusive

    train_indices = []
    val_indices = []
    test_indices = []

    for index, row in df.iterrows():
        if row['eff_dt_1']:
            if row['eff_dt_1'] <= train_date_upper:
                train_indices.append(index)
            elif row['eff_dt_1'] <= val_date_upper:
                val_indices.append(index)
            else:
                test_indices.append(index)

    X_train = X.loc[train_indices]  # NOTE: probably broken if called from v1, since series have no loc fn
    X_val = X.loc[val_indices]
    X_test = X.loc[test_indices]

    y_train = y[train_indices]
    y_val = y[val_indices]
    y_test = y[test_indices]

    return (X_train, X_val, X_test, y_train, y_val, y_test), (train_indices, val_indices, test_indices)


def word_tokenize(s):
    tokens = s.split(' ')
    if tokens[-1] == '':
        tokens = tokens[:-1]
    return tokens


"""
Prepares datasets that are split by date
"""
def prep_dataset_v2(num_classes_train=-1, num_classes_predict=-1, augmented=False):
    if augmented:
        df = pd.read_pickle(COURSE_OUTCOME_LIST_FILE_AUGMENTED)
    else:
        df = pd.read_pickle(COURSE_OUTCOME_LIST_FILE)
    df['eff_dt_1'] = df['eff_dt_1'].apply(datetime.date.fromisoformat)

    y = df['ACAD_PLAN_1']
    X = df.loc[:, ['course_history', 'RELATIVE_TERM', 'CRSE_GRADE_INPUT']]

    (X_train, X_val, X_test, y_train, y_val, y_test), (train_indices, val_indices, test_indices) = date_split(X, y, df)

    if num_classes_train > 0:
        X_train['course_history'] = X_train['course_history'].apply(truncate_class, args=[num_classes_train])
        X_train['RELATIVE_TERM'] = X_train['RELATIVE_TERM'].apply(truncate_class, args=[num_classes_train])
        X_train['CRSE_GRADE_INPUT'] = X_train['CRSE_GRADE_INPUT'].apply(truncate_class, args=[num_classes_train])

    if num_classes_predict > 0:
        X_val['course_history'] = X_val['course_history'].apply(truncate_class, args=[num_classes_predict])
        X_val['RELATIVE_TERM'] = X_val['RELATIVE_TERM'].apply(truncate_class, args=[num_classes_predict])
        X_val['CRSE_GRADE_INPUT'] = X_val['CRSE_GRADE_INPUT'].apply(truncate_class, args=[num_classes_predict])

        X_test['course_history'] = X_test['course_history'].apply(truncate_class, args=[num_classes_predict])
        X_test['RELATIVE_TERM'] = X_test['RELATIVE_TERM'].apply(truncate_class, args=[num_classes_predict])
        X_test['CRSE_GRADE_INPUT'] = X_test['CRSE_GRADE_INPUT'].apply(truncate_class, args=[num_classes_predict])

    return X_train, X_val, X_test, y_train, y_val, y_test


"""
DEPRECATED
Prepares dataset. Uses first dataset, which has now since been replaced
"""
def prep_dataset_v1(num_classes_predict=-1, vectorize=False):
    df = pd.read_feather(COURSE_MAJOR_FILE, use_threads=True)

    if vectorize:
        vectorizer, X = vectorize_course_history(df.loc[:, 'course_history'])
    else:
        X = df.loc[:, 'course_history']
    y = df['DEGREE_1']

    (X_train, X_val, X_test, y_train, y_val, y_test), (train_indices, val_indices, test_indices) = date_split(X, y, df)

    if num_classes_predict > 0:
        df['course_history'] = df['course_history'].apply(truncate_class, args=[num_classes_predict])

    if vectorize:
        _, X_truncated = vectorize_course_history(df.loc[:, 'course_history'], vectorizer=vectorizer)
    else:
        X_truncated = df.loc[:, 'course_history']

    X_val = X_truncated[val_indices]
    X_test = X_truncated[test_indices]

    y_val = y_val[val_indices]
    y_test = y_test[test_indices]

    return X_train, X_val, X_test, y_train, y_val, y_test


def subtokenize_single_course(course_str):
    match = re.match(r"([^0-9]+)([0-9]+)([^0-9]*)", course_str, re.I)
    items = list(match.groups())
    if items[-1] == "":
        items = items[:-1]
    if len(items[1]) == 3:
        items.insert(1, items[1][0])
        items[2] = items[2][1:]
    return items


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


"""
y is list of true labels
y_pred is list of lists of top_n predictions
returns: a list y_top_n, a conversion of y_pred where if y_pred contains a correct
         prediction, chooses said prediction. Else chooses arbitrary wrong prediction.
"""
def top_n_conversion(y, y_pred):
    y_top_n = []
    for i in range(len(y)):
        if y[i] in y_pred[i]:
            y_top_n.append(y[i].item())
        else:
            y_top_n.append(y_pred[i][0])
    return y_top_n


"""
takes in list of majors and outputs list of degree categories
"""
def degrees_to_categories(y):
    categories_df = pd.read_csv(DEGREE_CATEGORY_FILE)
    categories_dict = pd.Series(categories_df.DEGREE_CATEGORY.values, index=categories_df.DEGREE).to_dict()
    y_categories = [degrees_to_categories_single(c, categories_dict) for c in y]
    return y_categories


def degrees_to_categories_single(c, dict):
    if c in dict and dict[c] in CATEGORY_LIST:
        return dict[c]
    else:
        return 'OTHER'


"""
Bias testing helper functions
"""
def get_bias_datasets(test=False):
    if not test:
        gender_stem_df = pd.read_pickle(GENDER_STEM_STEREOTYPE_VAL_FILE)
        gender_stem_anti_df = pd.read_pickle(GENDER_STEM_ANTI_STEREOTYPE_VAL_FILE)
        gpa_stem_df = pd.read_pickle(GPA_STEM_STEREOTYPE_VAL_FILE)
        gpa_stem_anti_df = pd.read_pickle(GPA_STEM_ANTI_STEREOTYPE_VAL_FILE)

        male_df = pd.read_pickle(MALE_VAL_FILE)
        female_df = pd.read_pickle(FEMALE_VAL_FILE)
        high_gpa_df = pd.read_pickle(HIGH_GPA_VAL_FILE)
        low_gpa_df = pd.read_pickle(LOW_GPA_VAL_FILE)
    else:
        gender_stem_df = pd.read_pickle(GENDER_STEM_STEREOTYPE_TEST_FILE)
        gender_stem_anti_df = pd.read_pickle(GENDER_STEM_ANTI_STEREOTYPE_TEST_FILE)
        gpa_stem_df = pd.read_pickle(GPA_STEM_STEREOTYPE_TEST_FILE)
        gpa_stem_anti_df = pd.read_pickle(GPA_STEM_ANTI_STEREOTYPE_TEST_FILE)

        male_df = pd.read_pickle(MALE_TEST_FILE)
        female_df = pd.read_pickle(FEMALE_TEST_FILE)
        high_gpa_df = pd.read_pickle(HIGH_GPA_TEST_FILE)
        low_gpa_df = pd.read_pickle(LOW_GPA_TEST_FILE)

    return gender_stem_df, gender_stem_anti_df, gpa_stem_df, gpa_stem_anti_df, male_df, female_df, high_gpa_df, low_gpa_df


def evaluate_model_bias(model, args, evaluation_fn, num_classes_predict=0, categories=False, top_n=1, test=False):
    gender_stem_df, gender_stem_anti_df, gpa_stem_df, gpa_stem_anti_df, male_df, female_df, high_gpa_df, low_gpa_df = get_bias_datasets(test=test)

    gender_stem_report = evaluation_fn(model, gender_stem_df, args, num_classes_predict=num_classes_predict, categories=categories, top_n=top_n)
    gender_stem_anti_report = evaluation_fn(model, gender_stem_anti_df, args, num_classes_predict=num_classes_predict, categories=categories, top_n=top_n)
    gpa_stem_report = evaluation_fn(model, gpa_stem_df, args, num_classes_predict=num_classes_predict, categories=categories, top_n=top_n)
    gpa_stem_anti_report = evaluation_fn(model, gpa_stem_anti_df, args, num_classes_predict=num_classes_predict, categories=categories, top_n=top_n)

    male_report = evaluation_fn(model, male_df, args, num_classes_predict=num_classes_predict, categories=categories, top_n=top_n)
    female_report = evaluation_fn(model, female_df, args, num_classes_predict=num_classes_predict, categories=categories, top_n=top_n)
    high_gpa_report = evaluation_fn(model, high_gpa_df, args, num_classes_predict=num_classes_predict, categories=categories, top_n=top_n)
    low_gpa_report = evaluation_fn(model, low_gpa_df, args, num_classes_predict=num_classes_predict, categories=categories, top_n=top_n)

    print(f"Macro f1-score for Gender-STEM stereotype dataset: {gender_stem_report['macro avg']['f1-score']}")
    print(f"Macro f1-score for Gender-STEM anti stereotype dataset: {gender_stem_anti_report['macro avg']['f1-score']}")
    print(f"Macro f1-score for GPA-STEM stereotype dataset: {gpa_stem_report['macro avg']['f1-score']}")
    print(f"Macro f1-score for GPA-STEM anti-stereotype dataset: {gpa_stem_anti_report['macro avg']['f1-score']}")

    print(f"Macro f1-score for male dataset: {male_report['macro avg']['f1-score']}")
    print(f"Macro f1-score for female dataset: {female_report['macro avg']['f1-score']}")
    print(f"Macro f1-score for high GPA dataset: {high_gpa_report['macro avg']['f1-score']}")
    print(f"Macro f1-score for low GPA dataset: {low_gpa_report['macro avg']['f1-score']}")


"""
Evaluation helper fn to be used on sklearn models
"""
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


def main():
    print(get_vocab())

if __name__ == '__main__':
    main()
