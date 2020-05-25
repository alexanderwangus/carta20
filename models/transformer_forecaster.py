import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import util
from gensim.models import Word2Vec
from course_embeddings.course2vec import get_course2vec_model_path
from deep_course2vec import subtokenize_features, get_X_lens_v2, train_model, evaluate_model
import torch
import torch.nn as nn
import numpy as np
import torchtext
import itertools
import copy


TRAIN_LENGTH = 10
PREDICT_LENGTH = 10


class TransformerForecaster(nn.Module):
    def __init__(self, embed_size, vocab_sizes, num_classes, num_layers=3, num_heads=5, dropout=0.2, dim_feedforward=128):
        super(TransformerForecaster, self).__init__()

        n_course_tokens, n_term_tokens, n_grade_tokens = vocab_sizes
        self.course_embedder = nn.Embedding(n_course_tokens, embed_size)
        self.term_embedder = nn.Embedding(n_term_tokens, embed_size)
        self.grade_embedder = nn.Embedding(n_grade_tokens, embed_size)

        encoder_layers = nn.TransformerEncoderLayer(3 * embed_size, num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.decoder = nn.Linear(3 * 3 *  embed_size, num_classes)

        self.init_weights()


    def init_weights(self):
        initrange = 0.1
        self.course_embedder.weight.data.uniform_(-initrange, initrange)
        self.grade_embedder.weight.data.uniform_(-initrange, initrange)
        self.term_embedder.weight.data.uniform_(-initrange, initrange)

        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)


    def forward(self, sentences, X_lens):
        # sentences: (batch, max_seq_len, 3)
        sentences = sentences.long()
        course_sentences = sentences[:, :, 0]   # (batch, max_seq_len)
        term_sentences = sentences[:, :, 1]
        grade_sentences = sentences[:, :, 2]

        max_len = sentences.size(1)

        idx = torch.arange(max_len)[None, :]  # (1, max_seq_len)
        if torch.cuda.is_available():
            idx = idx.cuda()
        lens_expanded = X_lens[:, None]  # (batch, 1)
        mask = idx >= lens_expanded
        # print(mask.size())

        course_output = self.course_embedder(course_sentences)  # (batch, max_seq_len, embed_size)
        grade_output = self.grade_embedder(grade_sentences)
        term_output = self.term_embedder(term_sentences)

        output = torch.cat([course_output, grade_output, term_output], dim=2)
        output = output.transpose(0, 1)

        output = self.encoder(output, src_key_padding_mask=mask)  # (max_seq_len, batch, 3 * embed_size)
        output_max, _ = torch.max(output, dim=0)
        output_min, _ = torch.min(output, dim=0)
        output = torch.cat([torch.mean(output, dim=0), output_max, output_min], dim=1)
        output = self.decoder(output)

        return output


    def predict(self, sentence, X_lens):
        out = self.forward(sentence, X_lens)
        predicted = torch.argmax(out, 1)
        if torch.cuda.is_available():
            predicted = [t.item() for t in predicted]
        return predicted


def dummy_tokenizer(l):
    return l


def get_torchtext(X, tokenizer):
    torchtext_data = torchtext.data.Field(tokenize=tokenizer)
    torchtext_data.build_vocab(X)
    return torchtext_data


def numericalize_data(X, course_torchtext, term_torchtext, grade_torchtext):
    X['course_history'] = X['course_history'].apply(lambda s: [course_torchtext.vocab.stoi[c] for c in s])
    X['RELATIVE_TERM'] = X['RELATIVE_TERM'].apply(lambda s: [term_torchtext.vocab.stoi[c] for c in s])
    X['CRSE_GRADE_INPUT'] = X['CRSE_GRADE_INPUT'].apply(lambda s: [grade_torchtext.vocab.stoi[c] for c in s])

    return X


def pad_data(course_list, max_length):
    vec = np.array(course_list)
    padding = 0 if len(vec) >= max_length else max_length - len(vec)
    vec = np.pad(vec, ((0, padding)), "constant", constant_values=0)[:max_length]
    return np.array(vec)


def featurize_data(X, course_torchtext, term_torchtext, grade_torchtext, max_length):
    X = numericalize_data(X, course_torchtext, term_torchtext, grade_torchtext)

    X_course_history = X["course_history"].apply(pad_data, args=[max_length])
    X_term = X["RELATIVE_TERM"].apply(pad_data, args=[max_length])
    X_grade = X["CRSE_GRADE_INPUT"].apply(pad_data, args=[max_length])

    X_course_history = np.stack(X_course_history.values)[..., np.newaxis]  # (num_students, max_length, 1)
    X_term =  np.stack(X_term.values)[..., np.newaxis]
    X_grade =  np.stack(X_grade.values)[..., np.newaxis]

    return np.concatenate([X_course_history, X_term, X_grade], axis=2)


def prep_data(num_classes_train=-1, num_classes_predict=-1, subtokenize=False, augment=False):
    _, X_val, X_test, _, y_val, y_test = util.prep_dataset_v2(num_classes_train=num_classes_train, num_classes_predict=num_classes_predict, augmented=augment)
    X_train, _, _, y_train, _, _ = util.prep_dataset_v2(num_classes_train=num_classes_train, num_classes_predict=num_classes_predict, augmented=False)


    X_train_lens = get_X_lens_v2(X_train, TRAIN_LENGTH)
    X_val_lens = get_X_lens_v2(X_val, PREDICT_LENGTH)

    if subtokenize:
        X_train = subtokenize_features(X_train)
        X_val = subtokenize_features(X_val)

    course_torchtext = get_torchtext(X_train["course_history"], dummy_tokenizer)
    n_course_tokens = len(course_torchtext.vocab.stoi)

    term_torchtext = get_torchtext(X_train["RELATIVE_TERM"], dummy_tokenizer)
    n_term_tokens = len(term_torchtext.vocab.stoi)

    grade_torchtext = get_torchtext(X_train["CRSE_GRADE_INPUT"], dummy_tokenizer)
    n_grade_tokens = len(grade_torchtext.vocab.stoi)

    X_train = featurize_data(X_train, course_torchtext, term_torchtext, grade_torchtext, TRAIN_LENGTH)
    X_val = featurize_data(X_val, course_torchtext, term_torchtext, grade_torchtext, PREDICT_LENGTH)

    y_train = y_train.values
    y_val = y_val.values

    return (X_train, X_train_lens, y_train, X_val, X_val_lens, y_val), (n_course_tokens, n_term_tokens, n_grade_tokens)


def run_transformer_forecaster(pretrained_transformer=False, training_set=None, num_classes_train=-1, num_classes_predict=-1, subtokenize=False, augment=False):
    print(f"\nRunning transformer with num_classes_train={num_classes_train}, num_classes_predict={num_classes_predict}")
    print(f"subtokenize = {subtokenize}, augmentation = {augment}")

    data, num_tokens = prep_data(num_classes_train=num_classes_train, num_classes_predict=num_classes_predict, subtokenize=subtokenize, augment=augment)

    batch_size = 32
    epochs = 30

    num_layers = 6
    num_heads = 4
    vec_size = 128
    dropout=0.2
    dim_feedforward=2048
    lr = 0.0001


    transformer_model_path = get_transformer_model_path(vec_size, batch_size, num_layers, num_heads, lr, dropout, dim_feedforward)

    if pretrained_transformer:
        print(f"Loading pretrained transformer state_dict from '{transformer_model_path}'")
        transformer_model = TransformerForecaster(vec_size, num_tokens, \
            util.NUM_CLASSES, num_layers=num_layers, num_heads=num_heads, dropout=dropout, dim_feedforward=dim_feedforward)
        transformer_model.load_state_dict(torch.load(transformer_model_path))
    else:
        transformer_model = train_transformer(epochs, data, vec_size, batch_size, num_layers, num_heads, lr,\
            num_tokens, dropout, dim_feedforward)

        print(f"Saving transformer to '{transformer_model_path}'")
        with open(transformer_model_path, 'wb') as f:
            torch.save(transformer_model.state_dict(), f)

    X_train, X_train_lens, y_train, X_val, X_val_lens, y_val = data
    val_results = evaluate_model(X_val, X_val_lens, y_val, transformer_model, ouput_dict=False)
    print(val_results)


def hyperparam_search(pretrained_transformer=False, training_set=None, num_classes_train=TRAIN_LENGTH, num_classes_predict=PREDICT_LENGTH, subtokenize=False, augment=False):
    print(f"\nRunning hyperparam search with num_classes_train={num_classes_train}, num_classes_predict={num_classes_predict}")
    print(f"subtokenize = {subtokenize}, augmentation = {augment}")

    data, num_tokens = prep_data(num_classes_train=num_classes_train, num_classes_predict=num_classes_predict, subtokenize=subtokenize, augment=augment)
    X_train, X_train_lens, y_train, X_val, X_val_lens, y_val = data

    batch_size = 32
    epochs = 30

    batch_sizes = [8, 16, 32, 64, 128]
    num_layers = [6]
    num_heads = [4]
    vec_size = [128]
    dropout=[0.2, 0.3]
    dim_feedforward=[1024, 2048, 4128]
    lrs = [0.00001, 0.00005, 0.0001]

    best_metric = -1
    best_config = {}
    best_model = None

    for hyperparams in itertools.product(num_layers, num_heads, vec_size, dropout, dim_feedforward, lrs, batch_sizes):
        nl = hyperparams[0]
        nh = hyperparams[1]
        vs = hyperparams[2]
        dp = hyperparams[3]
        d_ff = hyperparams[4]
        lr = hyperparams[5]
        bs = hyperparams[6]

        config = {"num_layers": nl, "num_heads": nh, "vec_size": vs, "dropout": dp, "dim_feedforward": d_ff, "lr": lr, "batch_size": bs}
        print(f"Running trial with {config}")

        transformer_model = train_transformer(epochs, data, vs, bs, nl, nh, lr,\
            num_tokens, dp, d_ff, verbose=False)

        metric = evaluate_model(X_val, X_val_lens, y_val, transformer_model, ouput_dict=True)['macro avg']['f1-score']
        print(f"Achieved metric of {metric}.")

        if metric > best_metric:
            print(f"New best metric of {metric} to beat old metric of {best_metric} found.")
            best_metric = metric
            best_config = copy.deepcopy(config)
            best_model = copy.deepcopy(transformer_model)
        print("\n")

    print(f"Best config found: {best_config}")
    transformer_model_path = get_transformer_model_path(vs, bs, nl, nh, lr, dp, d_ff)
    print(f"Saving transformer to '{transformer_model_path}'")
    with open(transformer_model_path, 'wb') as f:
        torch.save(best_model.state_dict(), f)


def train_transformer(epochs, data, vec_size, batch_size, num_layers, num_heads, lr, num_tokens, dropout, dim_feedforward, verbose=True):
    X_train, X_train_lens, y_train, X_val, X_val_lens, y_val = data

    transformer_model = TransformerForecaster(vec_size, num_tokens, \
        util.NUM_CLASSES, num_layers=num_layers, num_heads=num_heads, dropout=dropout, dim_feedforward=dim_feedforward)

    if verbose:
        print(f"Training transformer")
    transformer_model = train_model(transformer_model, X_train, X_train_lens, y_train, X_val, X_val_lens, y_val, \
        epochs, batch_size, lr, verbose=verbose)

    return transformer_model


def get_transformer_model_path(input_size, batch_size, num_layers, num_heads, lr, dropout, dim_ff):
    return f"transformer_saved_models/dim{input_size}_batch{batch_size}_layers{num_layers}_heads{num_heads}_lr{lr}_seq_len{TRAIN_LENGTH}_dropout{dropout}_dimff_{dim_ff}.model"


def main():
    hyperparam_search(subtokenize=True, augment=True, pretrained_transformer=False, training_set=None, num_classes_train=TRAIN_LENGTH, num_classes_predict=PREDICT_LENGTH)


if __name__ == '__main__':
    main()

"""
TODO:
- masking
- normal results
- augmented results
- subtokenization results

Running trial with {'num_layers': 1, 'num_heads': 4, 'vec_size': 64, 'dropout': 0.2, 'dim_feedforward': 1024, 'lr': 0.0005}
Achieved metric of 0.0947345292136418.
New best metric of 0.0947345292136418 to beat old metric of 0.08644299260486957 found.


 {'num_layers': 1, 'num_heads': 4, 'vec_size': 64, 'dropout': 0.3, 'dim_feedforward': 2048, 'lr': 0.0005}
"""
