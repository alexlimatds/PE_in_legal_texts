import pandas as pd
import numpy as np
import random, torch, transformers, time
from datetime import datetime
import functions
from sentence_transformers import SentenceTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

def evaluate_sbert(get_features, model_reference):
    encoder_id = 'sentence-transformers/LaBSE'
    n_iterations = 5
    
    # random seeds
    seeds = [(42 + i * 10) for i in range(n_iterations)]
    
    # loading sentences
    train_dir = 'train/'
    test_dir = 'test/'
    dic_docs_train = functions.read_docs(train_dir)
    dic_docs_test = functions.read_docs(test_dir)
    
    # generating features
    sent_encoder = SentenceTransformer(encoder_id)
    sent_encoder.max_seq_length = 512
    train_features, train_targets, docs_train_features = get_features(dic_docs_train, sent_encoder)
    test_features, test_targets, docs_test_features = get_features(dic_docs_test, sent_encoder)
    
    # training and evaluation
    eval_metrics = {}
    functions.evaluation_sbert(default_mlp_trainer, train_features, train_targets, test_features, test_targets, seeds, eval_metrics)
    functions.evaluation_sbert(svm_trainer, train_features, train_targets, test_features, test_targets, seeds, eval_metrics)
    functions.evaluation_sbert(lr_trainer, train_features, train_targets, test_features, test_targets, seeds, eval_metrics)
    functions.evaluation_sbert(naive_bayes_trainer, train_features, train_targets, test_features, test_targets, seeds, eval_metrics)
    
    # report
    metrics_df = pd.DataFrame(columns=['Precision', 'P std', 'Recall', 'R std', 'F1', 'F1 std'])
    for model_name, metrics in eval_metrics.items():
        metrics_df.loc[model_name] = [
            metrics['precision'], metrics['precision_std'], 
            metrics['recall'], metrics['recall_std'], 
            metrics['f1'], metrics['f1_std']
        ]
    functions.save_report_sbert(model_reference, metrics_df, eval_metrics, './')
    
    
def naive_bayes_trainer(X, y, seed):
    nb = GaussianNB()
    return nb.fit(X, y)
    
def lr_trainer(X, y, seed):
    logreg = LogisticRegression(solver='sag', max_iter=200, random_state=seed)
    return logreg.fit(X, y)
    
def svm_trainer(X, y, seed):
    svm = LinearSVC(random_state=seed)
    return svm.fit(X, y)

def default_mlp_trainer(X, y, seed):
    mlp = MLPClassifier(early_stopping=True, random_state=seed)
    return mlp.fit(X, y)
