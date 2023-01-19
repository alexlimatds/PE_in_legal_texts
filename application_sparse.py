import pandas as pd
import numpy as np
import random, time
from datetime import datetime
import functions
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support

def evaluate_sparse(encoder, model_reference, n_docs=None):
    n_iterations = 1 # because we exploit only Naive Bayes classifier it's not necessary to run many executions
    
    # random seeds
    seeds = [(42 + i * 10) for i in range(n_iterations)]
    
    # loading sentences
    train_dir = 'train/'
    test_dir = 'test/'
    dic_docs_train = functions.read_docs(train_dir)
    dic_docs_test = functions.read_docs(test_dir)
    if n_docs is not None: # just for tests and to speed up the execution
        dic_docs_train = functions.get_dic_subset(dic_docs_train, n_docs)
        dic_docs_test = functions.get_dic_subset(dic_docs_test, n_docs)
    
    # generating features
    encoder.train(dic_docs_train)
    train_features, train_targets = encoder.encode(dic_docs_train)
    test_features, test_targets = encoder.encode(dic_docs_test)
    
    # training and evaluation
    eval_metrics = {}
    train_and_evaluate(naive_bayes_trainer, train_features, train_targets, test_features, test_targets, seeds, eval_metrics)
    
    # report
    metrics_df = pd.DataFrame(columns=['Precision', 'P std', 'Recall', 'R std', 'F1', 'F1 std'])
    for model_name, metrics in eval_metrics.items():
        metrics_df.loc[model_name] = [
            metrics['precision'], metrics['precision_std'], 
            metrics['recall'], metrics['recall_std'], 
            metrics['f1'], metrics['f1_std']
        ]
    save_report(model_reference, metrics_df, eval_metrics, encoder.get_features_info(), './')
        
def naive_bayes_trainer(X, y, seed):
    nb = GaussianNB()
    return nb.fit(X, y)

def save_report(model_reference, avg_metrics, cv_metrics, features_info, dest_dir):
    """
    Creates and save a report.
    Arguments:
        avg_metrics : A pandas Dataframe with the averaged metrics.
        cv_metrics : A dictionary of dictionaries containing the scores by model.
        features_info (string) : information about the feature set.
        dest_dir (string) : The directory where the report will be saved.
    """
    report = (
        'RESULTS REPORT\n'
        f'Model: {model_reference}\n'
        'Evaluation: test set (many random seeds)\n'
        f'Feature set info: {features_info}'
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

def train_and_evaluate(trainer, train_features, train_targets, test_features, test_targets, seeds, eval_metrics):
    print('### Evaluation ###')
    train_metrics = []
    test_metrics = []
    for i, seed_val in enumerate(seeds):
        print(f'Started iteration {i + 1}')
        random.seed(seed_val)
        np.random.seed(seed_val)
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
