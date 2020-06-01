import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import util
from gensim.models import Word2Vec
from course_embeddings.course2vec import get_course2vec_model_path
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import copy
from deep_course2vec import train_model, get_X_lens_v2, featurize_student_v2, evaluate_model

TRAIN_LENGTH = 5
PREDICT_LENGTH = 5
NUM_FEATURES = 3

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, num_classes, num_layers=1, hidden_size=150, dropout=0.2):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.input_size = input_size
        self.num_layers = num_layers
        self.dropout_rate = dropout

        self.lstm_1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear_1 = nn.Linear(hidden_size, num_classes)
        self.linear_2 = nn.Linear(hidden_size, num_classes)

        self.drop = nn.Dropout(p=dropout)
        # self.relu = nn.ReLU()

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
        # output = self.linear_2(self.drop(self.relu(output)))

        return output


    def predict(self, sentence, X_lens, top_n=1):
        out = self.forward(sentence, X_lens)
        softmax = nn.Softmax(dim=1)
        out = softmax(out)
        if top_n == 1:
            predicted = torch.argmax(out, 1)
            if torch.cuda.is_available():
                predicted = [t.item() for t in predicted]
            return predicted
        else:
            top_n_vals, top_n_indices = torch.topk(out, top_n, dim=1)
            if torch.cuda.is_available():
                top_n_indices = [t.tolist() for t in top_n_indices]
            return top_n_indices



def lstm_course2vec(vec_size, win_size, min_count, epochs, pretrained_lstm=False, training_set=None, num_classes_train=-1, num_classes_predict=-1):
    print(f"\nRunning lstm with vec_size={vec_size}, win_size={win_size}, min_count={min_count}, epochs={epochs}, num_classes_predict={num_classes_predict}")

    # set up hyperparams, load model
    course2vec_model_path = get_course2vec_model_path(vec_size, win_size, min_count, feature_type="course_history")

    if training_set:
        course2vec_model = train_course2vec(training_set, course2vec_model_path, vec_size, win_size, min_count, epochs=epochs)
    else:
        course2vec_model = Word2Vec.load(course2vec_model_path)

    # prep datasets
    course2vec_params = {"vec_size": vec_size, "win_size": win_size, "min_count": min_count}

    _, X_val, X_test, _, y_val, y_test = util.prep_dataset_v3(num_classes_train=num_classes_train, num_classes_predict=num_classes_predict, augmented=False)
    X_train, _, _, y_train, _, _ = util.prep_dataset_v3(num_classes_train=num_classes_train, num_classes_predict=num_classes_predict, augmented=False)

    X_train_lens = get_X_lens_v2(X_train, vec_size)
    X_train = featurize_student_v2(X_train, course2vec_params, num_classes_train, subtokenize=False)
    X_val_lens = get_X_lens_v2(X_val, vec_size)
    X_val = featurize_student_v2(X_val, course2vec_params, num_classes_predict, subtokenize=False)
    y_train = y_train.values
    y_val = y_val.values

    batch_size = 32
    num_layers = 1
    hidden_size = 150
    dropout=0.2
    lr = 0.001
    lstm_model_path = get_lstm_model_path(vec_size, batch_size, num_layers, hidden_size, lr, dropout)
    lstm_model = LSTMForecaster(vec_size * NUM_FEATURES, util.NUM_CLASSES, num_layers=num_layers, hidden_size=hidden_size, dropout=dropout)

    if pretrained_lstm:
        print(f"Loading pretrained lstm state_dict from '{lstm_model_path}'")
        lstm_model.load_state_dict(torch.load(lstm_model_path))
    else:
        print(f"Training lstm")
        lstm_model = train_model(lstm_model, X_train, X_train_lens, y_train, X_val, X_val_lens, y_val, \
            epochs, batch_size, lr)

        print(f"Saving lstm to '{lstm_model_path}'")
        with open(lstm_model_path, 'wb') as f:
            torch.save(lstm_model.state_dict(), f)

    val_results = evaluate_model(X_val, X_val_lens, y_val, lstm_model, ouput_dict=False)
    print(val_results)


def get_lstm_model_path(input_size, batch_size, num_layers, hidden_size, lr, dropout):
    return f"lstm_saved_models/augmented_2/dim{input_size}_batch{batch_size}_layers{num_layers}_hidden{hidden_size}_lr{lr}_seq_len{TRAIN_LENGTH}_drop{dropout}.model"


def main():
    vec_size=150
    win_size=10
    min_count=1
    epochs=30
    lstm_course2vec(vec_size, win_size, min_count, epochs, pretrained_lstm=False, training_set=None, num_classes_train=TRAIN_LENGTH, num_classes_predict=PREDICT_LENGTH)

if __name__ == '__main__':
    main()
