import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import util
from gensim.models import Word2Vec
from models.logistic_regression_model import train_log_reg
from sklearn.metrics import classification_report
from course_embeddings.course2vec import get_course2vec_model_path, train_model, featurize_student, create_training_set, get_course_vec
import torch
import torch.nn as nn
from torch import optim
import numpy as np

MAX_LENGTH = 128

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, num_classes, batch_size, num_layers=1, hidden_size=100, dropout=0):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout_rate = dropout

        self.lstm_1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear_1 = nn.Linear(hidden_size, num_classes)
        self.linear_2 = nn.Linear(hidden_size, num_classes)

        self.drop = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

        # TODO: better random initialization
        # nn.init.xavier_uniform(self.lstm_1.weight)
        # nn.init.xavier_uniform(self.linear_1.weight)

    def forward(self, sentences, X_lens, hidden=None, cell=None):
        # sentences of size (batch, max_seq_len, input_size)

        sentences = torch.nn.utils.rnn.pack_padded_sequence(sentences, X_lens, batch_first=True, enforce_sorted=False)
        output, (h_n, c_n) = self.lstm_1(sentences)  # h_n: (num_layers * num_directions, batch, hidden_size)
        h_n = h_n[-1]
        # h_n_permuted = h_n.permute(1, 0, 2)  # (batch, num_layers * num_directions, hidden_size)
        # h_n_permuted = self.drop(h_n_permuted)
        # lin_1_out = self.drop(self.relu(self.linear_1(h_n_permuted.view(-1, self.num_layers * self.hidden_size))))
        output = self.linear_1(self.drop(h_n))

        return output

    def predict(self, sentence, X_lens):
        out = self.forward(sentence, X_lens)
        predicted = torch.argmax(out, 1)
        return predicted


def courses2vecs(course_str, course2vec_model, vec_size, max_length):
    course_list = course_str.split(',')
    vec = np.array([get_course_vec(course2vec_model, word, vec_size) for word in course_list])
    padding = 0 if len(vec) >= max_length else max_length - len(vec)
    vec = np.pad(vec, ((0, padding), (0, 0)), "constant", constant_values=0)[:max_length]
    return np.array(vec)


def featurize_student(X, course2vec_model, vec_size, max_length=MAX_LENGTH):
    X = X.apply(courses2vecs, args=[course2vec_model, vec_size, max_length])
    return np.stack(X.values)


def get_seq_len(course_str, max_length):
    course_list = course_str.split(',')
    return min(max_length, len(course_list))


def get_X_lens(X, course2vec_model, vec_size, max_length=MAX_LENGTH):
    return [get_seq_len(seq, max_length) for _, seq in X.items()]


def train_lstm(course2vec_model, X_train, X_train_lens, y_train, X_val, X_val_lens, y_val, vec_size, epochs, batch_size, num_layers, hidden_size, lr):
    model = LSTMForecaster(vec_size, util.NUM_CLASSES, batch_size, num_layers=num_layers, hidden_size=hidden_size)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # X_train = X_train[:1000]
    # y_train = y_train[:1000]
    # X_train_lens = X_train_lens[:1000]

    val_results = evaluate_model(X_val, X_val_lens, y_val, model)
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

            # if i//batch_size % 100 == 0:
            #     print(f"Epoch {epoch+1} of {epochs}, Batch {i//batch_size} of {len(X_train)//batch_size} ")

            optimizer.zero_grad()

            targets = torch.LongTensor([util.course_to_idx(course) for course in y_train[i:i+curr_batch_size]])
            sentences = torch.FloatTensor(X_train[i:i+curr_batch_size])
            probs = model(sentences, X_train_lens[i:i+curr_batch_size])
            loss = loss_function(probs, targets)
            loss.backward()
            optimizer.step()


        train_results = evaluate_model(X_train, X_train_lens, y_train, model)
        print(f"== Epoch {epoch+1} train results (macro avg) ==")
        print(f"  Precision: {train_results['macro avg']['precision']}")
        print(f"  Recall: {train_results['macro avg']['recall']}")
        print(f"  f1-score: {train_results['macro avg']['f1-score']}")

        val_results = evaluate_model(X_val, X_val_lens, y_val, model)
        print(f"== Epoch {epoch+1} val results (macro avg) ==")
        print(f"  Precision: {val_results['macro avg']['precision']}")
        print(f"  Recall: {val_results['macro avg']['recall']}")
        print(f"  f1-score: {val_results['macro avg']['f1-score']}\n")

        if val_results['macro avg']['f1-score'] > best_metric:
            print(f"New best model found with f1-score {val_results['macro avg']['f1-score']} beating previous value of {best_metric}")
            best_model = model
            best_metric = val_results['macro avg']['f1-score']

    return best_model

def evaluate_model(X, X_lens, y, model, ouput_dict=True):
    model.eval()

    with torch.no_grad():
        y = torch.LongTensor([util.course_to_idx(course) for course in y])
        sentences = torch.FloatTensor(X)
        y_pred = model.predict(sentences, X_lens)
        # print(y)
        # print(y_pred)
        # print(classification_report(y, y_pred, zero_division=0))
        return classification_report(y, y_pred, zero_division=0, output_dict=ouput_dict)



def lstm_course2vec(vec_size, win_size, min_count, epochs, pretrained_lstm=False, training_set=None, num_classes_predict=-1):
    print(f"\nRunning lstm with vec_size={vec_size}, win_size={win_size}, min_count={min_count}, epochs={epochs}, num_classes_predict={num_classes_predict}")

    # set up hyperparams, load model
    course2vec_model_path = get_course2vec_model_path(vec_size, win_size, min_count, feature_type="course_history")

    if training_set:
        course2vec_model = train_model(training_set, course2vec_model_path, vec_size, win_size, min_count, epochs=epochs)
    else:
        course2vec_model = Word2Vec.load(course2vec_model_path)

    # prep datasets
    X_train, X_val, X_test, y_train, y_val, y_test = util.prep_dataset(vectorize=False, num_classes_predict=num_classes_predict)
    X_train_lens = get_X_lens(X_train, course2vec_model, vec_size)
    X_train = featurize_student(X_train, course2vec_model, vec_size)
    X_val_lens = get_X_lens(X_val, course2vec_model, vec_size)
    X_val = featurize_student(X_val, course2vec_model, vec_size)
    y_train = y_train.values
    y_val = y_val.values

    batch_size = 32
    num_layers = 1
    hidden_size = 50
    lr = 0.001
    lstm_model_path = get_lstm_model_path(vec_size, batch_size, num_layers, hidden_size, lr)

    if pretrained_lstm:
        print(f"Loading pretrained lstm state_dict from '{lstm_model_path}'")
        lstm_model = LSTMForecaster(vec_size, util.NUM_CLASSES, batch_size, num_layers=num_layers, hidden_size=hidden_size)
        lstm_model.load_state_dict(torch.load(lstm_model_path))
    else:
        print(f"Training lstm")
        lstm_model = train_lstm(course2vec_model, X_train, X_train_lens, y_train, X_val, X_val_lens, y_val, \
            vec_size=vec_size, epochs=epochs, batch_size=batch_size, num_layers=num_layers, hidden_size=hidden_size, lr=lr)

        print(f"Saving lstm to '{lstm_model_path}'")
        with open(lstm_model_path, 'wb') as f:
            torch.save(lstm_model.state_dict(), f)

    # train and predict using logistic regression model
    # train_score, log_reg_model = train_log_reg(list(X_train), y_train)
    # print(f"train_score: {train_score}")
    # y_pred = log_reg_model.predict(list(X_val))
    # macro_f1 = classification_report(y_val, y_pred, output_dict=True, zero_division=0)['macro avg']['f1-score']
    # print(classification_report(y_val, y_pred, zero_division=0))
    # return macro_f1, model

    val_results = evaluate_model(X_val, X_val_lens, y_val, lstm_model, ouput_dict=False)
    print(val_results)


def get_lstm_model_path(input_size, batch_size, num_layers, hidden_size, lr):
    return f"lstm_saved_models/dim{input_size}_batch{batch_size}_layers{num_layers}_hidden{hidden_size}_lr{lr}_seq_len{MAX_LENGTH}.model"


def main():
    vec_size=150
    win_size=10
    min_count=1
    epochs=15
    lstm_course2vec(vec_size, win_size, min_count, epochs, pretrained_lstm=False, training_set=None)

if __name__ == '__main__':
    main()

"""
batch_size = 32
num_layers = 1
hidden_size = 50
lr = 0.001

vec_size=150, win_size=10, min_count=1, epochs=15, dropout=0
0.45

vec_size=150, win_size=10, min_count=1, epochs=20, dropout=0 (?)
0.481

majority guesser gets f1-score of 0.00174
"""
