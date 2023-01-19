import pandas as pd
import numpy as np
import random, torch, models_bert, transformers, time
from datetime import datetime
import functions

def evaluate_BERT(train_params):
    # setting labels
    labels_to_idx = {
        'Facts' : 0, 
        'Other' : 1
    }
    labels = [None] * len(labels_to_idx)
    for l, idx in labels_to_idx.items():
        labels[idx] = l
    train_params['n_classes'] = len(labels)
    train_params['positive_label'] = labels_to_idx['Facts']
    
    # tokenizer
    encoder_id = train_params['encoder_id']
    tokenizer = transformers.AutoTokenizer.from_pretrained(encoder_id)
    
    # loading sentences
    train_dir = 'train/'
    test_dir = 'test/'
    dic_docs_train = functions.read_docs(train_dir)
    dic_docs_test = functions.read_docs(test_dir)
    # dataset objects
    if train_params.get('n_documents') is not None: # used in tests to speed up the train procedure
        n_documents = train_params.get('n_documents')
        temp_docs_train = {k: dic_docs_train[k] for k in sorted(dic_docs_train.keys())[:n_documents]}
        temp_docs_test = {k: dic_docs_test[k] for k in sorted(dic_docs_test.keys())[:n_documents]}
        ds_train = models_bert.get_dataset(temp_docs_train, labels_to_idx, tokenizer)
        ds_test = models_bert.get_dataset(temp_docs_test, labels_to_idx, tokenizer)
    else:
        ds_train = models_bert.get_dataset(dic_docs_train, labels_to_idx, tokenizer)
        ds_test = models_bert.get_dataset(dic_docs_test, labels_to_idx, tokenizer)
    
    train_params['warmup'] = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    raw_metrics = {} # key: epoch, value: numpy tensor of shape (n_iterations, 5)
    confusion_matrices = {} # key: iteration_id, value: dictionary (key: epoch, value: confusion matrix)
    cv_start = time.perf_counter()
    seeds = [(42 + i * 10) for i in range(train_params['n_iterations'])]
    for i, seed_val in enumerate(seeds):
        print(f'Started iteration {i + 1}')
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        # train
        iteration_metrics, cm, train_time = models_bert.fit(train_params, ds_train, ds_test, device)
        confusion_matrices[i] = cm
        for epoch, scores in iteration_metrics.items():
            epoch_metrics = raw_metrics.get(epoch, None)
            if epoch_metrics is None:
                raw_metrics[epoch] = scores.reshape(1,-1)
            else:
                raw_metrics[epoch] = np.vstack((epoch_metrics, scores))
        print('  Iteration time: ', train_time)

    metrics = pd.DataFrame(columns=['Epoch', 'Train loss', 'std', 'Test loss', 'std', 'Precision', 'P std', 'Recall', 'R std', 'F1', 'F1 std'])
    for i, (epoch, scores) in enumerate(raw_metrics.items()):
        mean = np.mean(scores, axis=0)
        std = np.std(scores, axis=0)
        metrics.loc[i] = [
            f'{epoch}', 
            f'{mean[0]:.6f}', f'{std[0]:.6f}',    # train loss
            f'{mean[1]:.6f}', f'{std[1]:.6f}',    # test loss
            f'{mean[2]:.4f}', f'{std[2]:.4f}',    # precision
            f'{mean[3]:.4f}', f'{std[3]:.4f}',    # recall
            f'{mean[4]:.4f}', f'{std[4]:.4f}'     # f1
        ]
    
    cv_end = time.perf_counter()
    cv_time = time.strftime("%Hh%Mm%Ss", time.gmtime(cv_end - cv_start))
    print('End of evaluation. Total time:', cv_time)
    save_report(
        metrics, raw_metrics, labels, 
        confusion_matrices, 'test set (many random seeds)', train_params, cv_time, device, './'
    )

def save_report(
    avg_metrics, complete_metrics, labels, 
    confusion_matrices, evaluation, train_params, train_time, device, dest_dir):
    """
    Arguments:
        avg_metrics : A pandas Dataframe with the averaged metrics.
        complete_metrics : A dictionary with the metrics by epoch. The key indicates the epoch. 
                            Each value must be a numpy tensor of shape (n_iterations, 5).
        labels : list of all labels.
        confusion_matrices : A dictionary => key: iteration_id, value: dictionary (key: epoch, value: confusion matrix)
        evaluation : the kind of evalutaion (string). Cross-validation or Holdout.
        train_params : A dictionary.
        train_time : total time spent on training/evaluation (string).
        device : ID of GPU device
        dest_dir : The directory where the report will be saved.
    """
    model_reference = train_params['model_reference']
    report = (
        'RESULTS REPORT\n'
        f'Model: {model_reference}\n'
        f'Encoder: {train_params["encoder_id"] if not train_params["use_mock"] else "MOCK MODEL"}\n'
        f'Evaluation: {evaluation}\n'
        'Train scheme: fine-tuning\n'
        f'Batch size: {train_params["batch_size"]}\n'
        f'Dropout rate: {train_params["dropout_rate"]}\n'
        f'Learning rate: {train_params["learning_rate"]}\n'
        f'Adam Epsilon: {train_params["eps"]}\n'
        f'LR warmup: {train_params["warmup"]}\n'
        f'Use MLP: {train_params["use_mlp"]}\n'
        f'Use Normalization layer: {train_params["norm_layer"]}\n'
        f'Weight decay: {train_params["weight_decay"]}\n'
        f'Train time: {train_time}\n'
        f'GPU name: {torch.cuda.get_device_name(device)}\n'
    )
        
    memory_in_bytes = torch.cuda.get_device_properties(device).total_memory
    memory_in_gb = round((memory_in_bytes/1024)/1024/1024,2)
    report += f'GPU memory: {memory_in_gb}\n\n'
    
    report += 'Averages:\n'
    report += avg_metrics.to_string(index=False, justify='center')
    
    report += '\n\n*** Detailed report ***\n'
    
    report += f'\nConfusion matrices\n{"-"*18}\n'
    for i, label in enumerate(labels):
        report += f'{label}: {i} \n'
    for iteration_id, cm_dic in confusion_matrices.items():
        report += f'=> Iteration {iteration_id}:\n'
        for e, cm in cm_dic.items():
            report += f'Epoch {e}:\n{cm}\n'

    report += f'\nScores\n{"-"*6}\n'
    for epoch, scores in complete_metrics.items():
        df = pd.DataFrame(
            scores, 
            columns=['Train loss', 'Test loss', 'Precision', 'Recall', 'F1'], 
            index=[f'Iteration {i}' for i in range(scores.shape[0])])
        report += f'Epoch: {epoch}\n' + df.to_string() + '\n\n'
    
        
    with open(dest_dir + f'report-{model_reference}_ft_{datetime.now().strftime("%Y-%m-%d-%Hh%Mmin")}.txt', 'w') as f:
        f.write(report)
