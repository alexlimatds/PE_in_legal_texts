from os import listdir
import pandas as pd
import csv
from datetime import datetime
import random
import numpy as np
import torch
import sklearn
from sklearn.metrics import precision_recall_fscore_support

def read_docs(dir_path):
  """
  Read the docs in a directory.
  Params:
    dir_path (string): path to the directory that contains the documents.
  Returns:
    A dictionary whose keys are the names of the read files and the values are 
    pandas dataframes. Each dataframe has sentence and label columns.
  """
  docs = {} # key: file name, value: dataframe with sentences and labels
  for f in listdir(dir_path):
    df = pd.read_csv(
        dir_path + f, 
        sep='\t', 
        quoting=csv.QUOTE_NONE, 
        names=['sentence', 'label'])
    docs[f] = df
  return docs

def get_dic_subset(dic, n_entries):
    '''
    Returns a subset of a dictionary.
    Arguments:
        dic (dictionary): the dictionary from which extract a subset.
        n_entries (int): the number of entries of the subset.
    '''
    keys = list(dic.keys())
    assert n_entries <= len(keys)
    sub_dic = {}
    for i in range(n_entries):
        k = keys[i]
        sub_dic[k] = dic[k]
    return sub_dic

def save_report_sbert(model_reference, avg_metrics, cv_metrics, dest_dir):
    """
    Creates and save a report.
    Arguments:
        avg_metrics : A pandas Dataframe with the averaged metrics.
        cv_metrics : A dictionary of dictionaries containing the scores by model.
        dest_dir : The directory where the report will be saved.
    """
    report = (
        'RESULTS REPORT\n'
        f'Model: {model_reference}\n'
        'Evaluation: test set (many random seeds)\n'
    )
    
    report += '\nAverages:\n'
    report += avg_metrics.to_string(index=True, justify='center')
    
    report += '\n\n** Detailed report **\n'
    report += '\nScores:\n'
    for model, dict_scores in cv_metrics.items():
        scores = np.hstack((dict_scores['scores_train'], dict_scores['scores_test']))
        df = pd.DataFrame(
            scores,
            columns=['Train P', 'Train R', 'Train F1', 'Test P', 'Test R', 'Test F1'], 
            index=[f'Iteration {i}' for i in range(scores.shape[0])]
        )
        report += f'Model: {model}\n' + df.to_string() + '\n\n'
        
    with open(dest_dir + f'report-{model_reference}_{datetime.now().strftime("%Y-%m-%d-%Hh%Mmin")}.txt', 'w') as f:
        f.write(report)

def evaluation_sbert(trainer, train_features, train_targets, test_features, test_targets, seeds, eval_metrics):
    print('### Evaluation ###')
    train_metrics = []
    test_metrics = []
    for i, seed_val in enumerate(seeds):
        print(f'Started iteration {i + 1}')
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        #training model
        model = trainer(train_features, train_targets, seed_val)
        # evaluating model
        # test metrics
        predictions = model.predict(test_features)
        p_test, r_test, f1_test, _ = precision_recall_fscore_support(
            test_targets, 
            predictions, 
            average='binary', 
            pos_label='Facts', 
            zero_division=0
        )
        test_metrics.append([p_test, r_test, f1_test])
        # train metrics
        predictions = model.predict(train_features)
        p_train, r_train, f1_train, _ = precision_recall_fscore_support(
            train_targets, 
            predictions, 
            average='binary', 
            pos_label='Facts', 
            zero_division=0
        )
        train_metrics.append([p_train, r_train, f1_train])
    
    test_metrics = np.array(test_metrics) # dim 0: fold id, dim 1: metric (0: precision, 1: recall, 2: F1)
    test_mean = np.mean(test_metrics, axis=0)
    test_std = np.std(test_metrics, axis=0)
    train_metrics = np.array(train_metrics) # dim 0: fold id, dim 1: metric (0: precision, 1: recall, 2: F1)
    train_mean = np.mean(train_metrics, axis=0)
    train_std = np.std(train_metrics, axis=0)
    
    print(f'Mean precision - std deviation => train: {train_mean[0]:.4f} {train_std[0]:.4f} \t test: {test_mean[0]:.4f} {test_std[0]:.4f}')
    print(f'Mean recall - std deviation    => train: {train_mean[1]:.4f} {train_std[1]:.4f} \t test: {test_mean[1]:.4f} {test_std[1]:.4f}')
    print(f'Mean f1 - std deviation        => train: {train_mean[2]:.4f} {train_std[2]:.4f} \t test: {test_mean[2]:.4f} {test_std[2]:.4f}')

    eval_metrics[model.__class__.__name__] = {
        'precision': f'{test_mean[0]:.4f}', 
        'precision_std': f'{test_std[0]:.4f}', 
        'recall': f'{test_mean[1]:.4f}', 
        'recall_std': f'{test_std[1]:.4f}', 
        'f1': f'{test_mean[2]:.4f}', 
        'f1_std': f'{test_std[2]:.4f}', 
        'scores_train': train_metrics, 
        'scores_test': test_metrics
    }

def getPositionEncoding(seq_len, d, n=10000):
    """
    Returns a positional encoding matrix.
    Code adapted from https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
    Arguments:
        seq_len : the length of the sequence.
        d : the embedding / encoding dimension.
    Returns:
        A PyTorch tensor with shape (seq_len, d).
    """
    P = torch.zeros((seq_len, d))
    for k in range(seq_len):
        for i in torch.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P

def get_mvPE(seq_len, d, k, n=10000):
    """
    Returns a mvPE matrix.
    Code adapted from https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
    Arguments:
        seq_len : the length of the sequence.
        d : the embedding / encoding dimension.
        k : mvPE step
    Returns:
        A PyTorch tensor with shape (seq_len, d).
    """
    P = torch.zeros((seq_len, d))
    for t in range(seq_len):
        for i in torch.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[t, 2*i] = np.sin(t*k/denominator)
            P[t, 2*i+1] = np.cos(t*k/denominator)
    return P