import pandas as pd
import numpy as np
import random, torch, time
from datetime import datetime
import functions
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

def get_features(dic_docs, encoding_scheme):
    k = 3000 # mvPE step parameter
    features = None
    labels = []
    features_by_doc = {}
    
    # search for the maximum number of sentences in all documents to compute matrix just one time
    max_seq_len = 0
    for doc_id, df in dic_docs.items():
        max_seq_len = max(max_seq_len, df.shape[0])
    
    # PE/mvPE matrix
    embedding_dim = 768 # standard embedding dim of BERT and SBERT
    if encoding_scheme == 'PE':
        PE = functions.getPositionEncoding(max_seq_len, embedding_dim)
    elif encoding_scheme == 'mvPE':
        PE = functions.get_mvPE(max_seq_len, embedding_dim, k)
    
    for doc_id, df in dic_docs.items():
        sentences = df['sentence'].to_list()
        embedding = PE[0:len(sentences), :]
        features_by_doc[doc_id] = embedding
        if features is None:
            features = embedding
        else:
            features = torch.vstack((features, embedding))
        labels.extend(df['label'].tolist())
    for doc_id, tensor in features_by_doc.items():
        features_by_doc[doc_id] = tensor.detach().to('cpu').numpy()
    return features.detach().to('cpu').numpy(), labels, features_by_doc

def evaluate_pe(encoding_scheme):
    '''
    Arguments:
        encoding_scheme (string) : 'PE' or 'mvPE'
    '''
    assert encoding_scheme in ['PE', 'mvPE']
    n_iterations = 5
    
    # random seeds
    seeds = [(42 + i * 10) for i in range(n_iterations)]
    
    # loading sentences
    train_dir = 'train/'
    test_dir = 'test/'
    dic_docs_train = functions.read_docs(train_dir)
    dic_docs_test = functions.read_docs(test_dir)
    
    # generating features
    train_features, train_targets, docs_train_features = get_features(dic_docs_train, encoding_scheme)
    test_features, test_targets, docs_test_features = get_features(dic_docs_test, encoding_scheme)
    
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
    functions.save_report_sbert(encoding_scheme, metrics_df, eval_metrics, './')


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
