import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import util
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch import optim
import copy


"""
Evaluation helper fn to be used on pytorch models
"""
def evaluate_pytorch_model(X, X_lens, y, model, output_dict=True, categories=False, top_n=1):
    model.eval()
    batch_size = 32

    with torch.no_grad():
        if categories:
            y = torch.LongTensor([util.category_to_idx(course) for course in y])
        else:
            y = torch.LongTensor([util.course_to_idx(course) for course in y])

        y_pred = []
        for i in range(0, len(X), batch_size):
            curr_batch_size = batch_size
            if i + batch_size > len(X):
                curr_batch_size = len(X) - i + 1

            sentences = torch.FloatTensor(X[i:i+curr_batch_size])
            sentence_lens = torch.LongTensor(X_lens[i:i+curr_batch_size])
            if torch.cuda.is_available():
                sentences = sentences.cuda()
                sentence_lens = sentence_lens.cuda()
            y_pred += model.predict(sentences, sentence_lens, top_n=top_n)

        if top_n != 1:
            y_pred = top_n_conversion(y, y_pred)

        return classification_report(y, y_pred, zero_division=0, output_dict=output_dict)


"""
Training helper fn for pytorch models
"""
def train_model(model, X_train, X_train_lens, y_train, X_val, X_val_lens, y_val, epochs, batch_size, lr, verbose=True, categories=False, top_n=1):
    if torch.cuda.is_available():
        model = model.cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Training on {device}")

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    val_results = evaluate_pytorch_model(X_val, X_val_lens, y_val, model, categories=categories, top_n=top_n)
    if verbose:
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

            optimizer.zero_grad()

            if categories:
                targets = torch.LongTensor([util.category_to_idx(course) for course in y_train[i:i+curr_batch_size]])
            else:
                targets = torch.LongTensor([util.course_to_idx(course) for course in y_train[i:i+curr_batch_size]])

            sentences = torch.FloatTensor(X_train[i:i+curr_batch_size])
            sentence_lens = torch.LongTensor(X_train_lens[i:i+curr_batch_size])
            if torch.cuda.is_available():
                sentences = sentences.cuda()
                sentence_lens = sentence_lens.cuda()
                targets = targets.cuda()
            probs = model(sentences, sentence_lens)
            loss = loss_function(probs, targets)
            loss.backward()
            optimizer.step()

        if verbose:
            train_results = evaluate_pytorch_model(X_train, X_train_lens, y_train, model, categories=categories, top_n=top_n)
            print(f"== Epoch {epoch+1} train results (macro avg) ==")
            print(f"  Precision: {train_results['macro avg']['precision']}")
            print(f"  Recall: {train_results['macro avg']['recall']}")
            print(f"  f1-score: {train_results['macro avg']['f1-score']}")

        val_results = evaluate_pytorch_model(X_val, X_val_lens, y_val, model, categories=categories, top_n=top_n)
        if verbose:
            print(f"== Epoch {epoch+1} val results (macro avg) ==")
            print(f"  Precision: {val_results['macro avg']['precision']}")
            print(f"  Recall: {val_results['macro avg']['recall']}")
            print(f"  f1-score: {val_results['macro avg']['f1-score']}\n")

        if val_results['macro avg']['f1-score'] > best_metric:
            if verbose:
                print(f"New best model found with f1-score {val_results['macro avg']['f1-score']} beating previous value of {best_metric}")
            best_model = copy.deepcopy(model)
            best_metric = val_results['macro avg']['f1-score']

    return best_model
