import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import util
from gensim.models import Word2Vec
from course_embeddings.course2vec import get_course2vec_model_path
import torch
import torch.nn as nn
import numpy as np
from deep_course2vec import train_model, get_X_lens_v2, featurize_student_v2, evaluate_model

TRAIN_LENGTH = 64
PREDICT_LENGTH = 64
NUM_FEATURES = 3


class TransformerForecaster(nn.Module):
    def __init__(self, input_size, num_classes, num_layers=3, num_heads=5):
        super(TransformerForecaster, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(input_size, num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(3*input_size, num_classes)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, sentences, X_lens):
        # sentences of size (batch, max_seq_len, input_size)
        max_len = sentences.size(1)

        idx = torch.arange(max_len)[None, :, None]
        lens_expanded = X_lens[:, None, None].expand(sentences.size())  # (batch, max_seq_len, input_size)
        mask = idx >= lens_expanded
        sentences[mask] = 0

        output = self.encoder(sentences)  # (batch, max_seq_len, input_size)
        output_max, _ = torch.max(output, dim=1)
        output_min, _ = torch.min(output, dim=1)
        output = torch.cat([torch.mean(output, dim=1), output_max, output_min], dim=1)
        output = self.decoder(output)

        return output

    def predict(self, sentence, X_lens):
        out = self.forward(sentence, X_lens)
        predicted = torch.argmax(out, 1)
        return predicted


def transformer_course2vec(vec_size, win_size, min_count, epochs, pretrained_transformer=False, training_set=None, num_classes_train=-1, num_classes_predict=-1):
    print(f"\nRunning transformer with vec_size={vec_size}, win_size={win_size}, min_count={min_count}, epochs={epochs}, num_classes_predict={num_classes_predict}")

    # set up hyperparams, load model
    course2vec_model_path = get_course2vec_model_path(vec_size, win_size, min_count, feature_type="course_history")

    if training_set:
        course2vec_model = train_course2vec(training_set, course2vec_model_path, vec_size, win_size, min_count, epochs=epochs)
    else:
        course2vec_model = Word2Vec.load(course2vec_model_path)

    # prep datasets
    course2vec_params = {"vec_size": vec_size, "win_size": win_size, "min_count": min_count}

    _, X_val, X_test, _, y_val, y_test = util.prep_dataset_v2(num_classes_train=num_classes_train, num_classes_predict=num_classes_predict, augmented=False)
    X_train, _, _, y_train, _, _ = util.prep_dataset_v2(num_classes_train=num_classes_train, num_classes_predict=num_classes_predict, augmented=False)

    X_train_lens = get_X_lens_v2(X_train, vec_size, TRAIN_LENGTH)
    X_train = featurize_student_v2(X_train, course2vec_params, TRAIN_LENGTH, subtokenize=False)
    X_val_lens = get_X_lens_v2(X_val, vec_size, PREDICT_LENGTH)
    X_val = featurize_student_v2(X_val, course2vec_params, PREDICT_LENGTH, subtokenize=False)
    y_train = y_train.values
    y_val = y_val.values

    batch_size = 32
    num_layers = 6
    num_heads = 8
    lr = 0.001
    transformer_model_path = get_transformer_model_path(vec_size, batch_size, num_layers, lr, num_heads)
    transformer_model = TransformerForecaster(vec_size * NUM_FEATURES, util.NUM_CLASSES, num_layers=num_layers)

    if pretrained_transformer:
        print(f"Loading pretrained transformer state_dict from '{transformer_model_path}'")
        transformer_model.load_state_dict(torch.load(transformer_model_path))
    else:
        print(f"Training transformer")
        transformer_model = train_model(transformer_model, course2vec_model, X_train, X_train_lens, y_train, X_val, X_val_lens, y_val, \
            epochs, batch_size, lr)

        print(f"Saving transformer to '{transformer_model_path}'")
        with open(transformer_model_path, 'wb') as f:
            torch.save(transformer_model.state_dict(), f)

    val_results = evaluate_model(X_val, X_val_lens, y_val, transformer_model, ouput_dict=False)
    print(val_results)


def get_transformer_model_path(input_size, batch_size, num_layers, lr, num_heads):
    return f"transformer_saved_models/dim{input_size}_batch{batch_size}_layers{num_layers}_heads{num_heads}_lr{lr}_seq_len{TRAIN_LENGTH}.model"


def main():
    vec_size=150
    win_size=10
    min_count=1
    epochs=30
    transformer_course2vec(vec_size, win_size, min_count, epochs, pretrained_transformer=False, training_set=None, num_classes_train=TRAIN_LENGTH, num_classes_predict=PREDICT_LENGTH)


if __name__ == '__main__':
    main()
