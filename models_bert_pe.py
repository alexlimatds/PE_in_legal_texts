import torch
from torch.utils.data import Dataset
import transformers
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
import time

class SingleSC_PE_BERT(torch.nn.Module):
    """
    Single Sentence Classifier based on a BERT kind encoder and Positional Encoding.
    Single sentence means this classifier encodes each sentence in a 
    individual way, i.e, it doesn't take in account other sentences in the 
    same document.
    The sentence encoder must be a pre-trained model based on BERT's architecture 
    like BERT, RoBERTa and ALBERT.
    """
    def __init__(self, encoder_id, n_classes, dropout_rate, embedding_dim, use_mlp, pe_combination, n_hidden_mlp=100, norm_layer=False):
        '''
        This model comprises a pre-trained sentence encoder and a classification head. 
        The sentence encoder must be a model following BERT architecture.  
        The classification head is a linear classifier (a single feedforward layer).
        Arguments:
            encoder_id: ID (string) of the encoder model in Hugging Faces repository.
            n_classes: number of classes.
            dropout_rate: dropout rate of classification layer.
            embedding_dim: dimension of hidden units in the sentence encoder (e.g., 768 for BERT).
            use_mlp: indicates the use of a MLP classifier (True) or a single layer one (False) in the classification head.
            pe_combination: how sentence embeddings and PE must be combined: 'sum' or 'concatenation'.
            n_hidden_mlp: the number of hidden units of the MLP classifier.
            norm_layer: indicates if a normalization layer will be applied to the feature set before feeding the densen layer.
        '''
        super(SingleSC_PE_BERT, self).__init__()
        self.encoder = transformers.AutoModel.from_pretrained(encoder_id)
        self.n_classes = n_classes
        self.pe_combination = pe_combination
        if pe_combination == 'sum':
            combined_dim = embedding_dim
        elif pe_combination == 'concatenation':
            combined_dim = embedding_dim * 2
        else:
            raise ValueError('Invalid value for pe_combination arg:', pe_combination)
        dropout = torch.nn.Dropout(dropout_rate)
        sequence_layers = []
        if norm_layer:
            sequence_layers.append(torch.nn.LayerNorm(combined_dim))
        if use_mlp:
            dense_hidden = torch.nn.Linear(combined_dim, n_hidden_mlp)
            torch.nn.init.kaiming_uniform_(dense_hidden.weight)
            relu = torch.nn.ReLU()
            dense_out = torch.nn.Linear(n_hidden_mlp, n_classes)
            torch.nn.init.xavier_uniform_(dense_out.weight)
            sequence_layers.extend([dropout, dense_hidden, relu, dropout, dense_out])
            self.classifier = torch.nn.Sequential(*sequence_layers)
        else:
            dense_out = torch.nn.Linear(combined_dim, n_classes)
            torch.nn.init.xavier_uniform_(dense_out.weight)
            sequence_layers.extend([dropout, dense_out])
            self.classifier = torch.nn.Sequential(*sequence_layers)

    def forward(self, input_ids, attention_mask, pe_features):
        '''
        Each call to this method process a batch of sentences. Each sentence is 
        individually encoded. This means the encoder doesn't take in account 
        other sentences from the source document when it encodes a sentence. 
        The sentence encodings are enriched with PE features before classification.
        This method returns one logit tensor for each sentence in the batch.
        Arguments:
            input_ids : tensor of shape (batch_size, seq_len)
            attention_mask : tensor of shape (batch_size, seq_len)
            pe_features : tensor of shape ((batch_size, seq_len, embedding_dim)
        Returns:
            logits : tensor of shape (n of sentences in batch, n of classes)
        '''
        output_1 = self.encoder(
            input_ids=input_ids,             # input_ids.shape: (batch_size, seq_len)
            attention_mask=attention_mask    # attention_mask.shape: (batch_size, seq_len)
        )
        hidden_state = output_1.last_hidden_state  # hidden states of last encoder's layer => shape: (batch_size, seq_len, embedd_dim)
        embeddings = hidden_state[:, 0]        # hidden states of the CLS tokens from the last layer => shape: (batch_size, embedd_dim)
        
        if self.pe_combination == 'sum':
            embeddings = embeddings + pe_features
        elif self.pe_combination == 'concatenation':
            embeddings = torch.hstack((embeddings, pe_features))
        
        logits = self.classifier(embeddings)   # logits.shape: (batch_size, num of classes)

        return logits

class MockSC_BERT(torch.nn.Module):
    '''
    A mock of SingleSC_BERT. It's usefull to accelerate the validation
    of the training loop.
    '''
    def __init__(self, n_classes):
        super(MockSC_BERT, self).__init__()
        self.classifier = torch.nn.Linear(10, n_classes) # 10 => random choice

    def forward(self, input_ids, attention_mask, pe_features):
        batch_size = input_ids.shape[0]
        mock_data = torch.rand((batch_size, 10), device=input_ids.device)
        logits = self.classifier(mock_data)    # shape: (batch_size, n_classes)

        return logits
    
def getPositionEncoding(seq_len, d, k, n=10000):
    """
    Returns a maximum variances positional encoding (mvPE) matrix.
    Code adapted from https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
    Arguments:
        seq_len : the length of the sequence.
        d : the embedding / encoding dimension.
        k : (integer) step parameter from mvPE algorithm.
        n : maximum supported sequence length.
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
    
class Single_SC_PE_Dataset(Dataset):
    """
    A dataset object to be used together a SingleSC_PE_BERT model. 
    Each item of the dataset represents a sole sentence.
    """
    def __init__(self, dic_docs, labels_to_idx, tokenizer, embedding_dim, max_sentence_len, k):
        """
        Arguments:
            dic_docs :
            labels_to_idx : dictionary that maps each label (string) to a index (integer).
            tokenizer : the tokenizer to be used to split the sentences into inputs 
                of a SingleSC_BERT.
            embedding_dim : embedding dimension.
            max_sentence_len : maximum number of tokens in a sentence.
            k : (integer) step parameter from mvPE algorithm.
        """
        
        self.labels = []
        self.targets = []
        self.input_ids = []
        self.attention_masks = []
        self.pe_features = []
        
        # search for the maximum number of sentences in a document to compute PE matrix just one time
        max_seq_len = 0
        for doc_id, df in dic_docs.items():
            max_seq_len = max(max_seq_len, df.shape[0])
        # PE matrix
        PE = getPositionEncoding(max_seq_len, embedding_dim, k)

        for doc_id, df in dic_docs.items():
            sentences = df['sentence'].tolist()
            self.pe_features.append(PE[0:len(sentences), :])
            tok_data = tokenizer(
                sentences, 
                add_special_tokens=True,
                padding='max_length', 
                max_length=max_sentence_len,
                return_token_type_ids=False, 
                return_attention_mask=True, 
                truncation=True, 
                return_tensors='pt'
            )
            self.input_ids.append(tok_data['input_ids'])
            self.attention_masks.append(tok_data['attention_mask'])
            labels_in_doc = df['label'].tolist()
            self.labels.extend(labels_in_doc)

        self.input_ids = torch.vstack(self.input_ids)
        self.attention_masks = torch.vstack(self.attention_masks)
        self.pe_features = torch.vstack(self.pe_features)
        
        # targets
        for l in self.labels:
            self.targets.append(labels_to_idx[l])
        self.targets = torch.tensor(self.targets, dtype=torch.long)
        
        self.len = len(self.targets)

    def __getitem__(self, index):
        return {
            'ids': self.input_ids[index],          # shape: (seq_len)
            'mask': self.attention_masks[index],   # shape: (seq_len)
            'pe': self.pe_features[index],         # shape: (embedding_dim)
            'target': self.targets[index],         # shape: (1)
            'label': self.labels[index]            # shape: (1)
        }
    
    def __len__(self):
        return self.len

def count_labels(ds):
    """
    Returns the number of sentences by label for a provided Single_SC_PE_Dataset.
    Arguments:
        ds : a Single_SC_PE_Dataset.
    Returns:
       A dictionary mapping each label (string) to its number of sentences (integer).
    """
    count_by_label = {l: 0 for l in ds.labels}
    for l in ds.labels:
        count_by_label[l] = count_by_label[l] + 1
    return count_by_label

def evaluate(model, test_dataloader, loss_function, positive_label, device):
    """
    Evaluates a provided SingleSC model.
    Arguments:
        model: the model to be evaluated.
        test_dataloader: torch.utils.data.DataLoader instance containing the test data.
        loss_function: instance of the loss function used to train the model.
        positive_label: the positive label used to compute the scores.
        device: device where the model is located.
    Returns:
        eval_loss (float): the computed test loss score.
        precision (float): the computed test Precision score.
        recall (float): the computed test Recall score.
        f1 (float): the computed test F1 score.
        confusion_matrix: the computed test confusion matrix.
    """
    predictions = torch.tensor([]).to(device)
    y_true = torch.tensor([]).to(device)
    eval_loss = 0
    model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            pe = data['pe'].to(device)
            y_true_batch = data['target'].to(device)
            y_hat = model(ids, mask, pe)
            loss = loss_function(y_hat, y_true_batch)
            eval_loss += loss.item()
            predictions_batch = y_hat.argmax(dim=1)
            predictions = torch.cat((predictions, predictions_batch))
            y_true = torch.cat((y_true, y_true_batch))
        predictions = predictions.detach().to('cpu').numpy()
        y_true = y_true.detach().to('cpu').numpy()
    eval_loss = eval_loss / len(test_dataloader)
    t_metrics = precision_recall_fscore_support(
        y_true, 
        predictions, 
        average='binary', 
        pos_label=positive_label,
        zero_division=0
    )
    cm = confusion_matrix(
        y_true, 
        predictions
    )
    
    return eval_loss, t_metrics[0], t_metrics[1], t_metrics[2], cm

def fit(train_params, ds_train, ds_test, device):
    """
    Creates and train an instance of SingleSC_BERT.
    Arguments:
        train_params: dictionary storing the training params.
        ds_train: instance of Single_SC_Dataset storing the train data.
        tokenizer: the tokenizer of the chosen pre-trained sentence encoder.
        device: device where the model is located.
    """
    learning_rate = train_params['learning_rate']
    weight_decay = train_params['weight_decay']
    n_epochs = train_params['n_epochs']
    batch_size = train_params['batch_size']
    encoder_id = train_params['encoder_id']
    n_classes = train_params['n_classes']
    dropout_rate = train_params['dropout_rate']
    embedding_dim = train_params['embedding_dim']
    use_mlp = train_params['use_mlp']
    use_mock = train_params['use_mock']
    
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=True)
    
    if use_mock:
        sentence_classifier = MockSC_BERT(n_classes).to(device)
    else:
        sentence_classifier = SingleSC_PE_BERT(
            encoder_id, 
            n_classes, 
            dropout_rate, 
            embedding_dim, 
            use_mlp, 
            train_params['combination']
        ).to(device)
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(
        sentence_classifier.parameters(), 
        lr=learning_rate, 
        betas=(0.9, 0.999), 
        eps=1e-8, 
        weight_decay=weight_decay
    )
    lr_scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = 0, 
        num_training_steps = len(dl_train) * n_epochs
    )
    
    metrics = {} # key: epoch number, value: numpy tensor storing train loss, test loss, Precision, Recall, F1
    confusion_matrices = {} # key: epoch number, value: scikit-learn's confusion matrix
    start_train = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        print(f'Starting epoch {epoch}... ', end='')
        start_epoch = time.perf_counter()
        epoch_loss = 0
        sentence_classifier.train()
        for train_data in dl_train:
            optimizer.zero_grad()
            ids = train_data['ids'].to(device)
            mask = train_data['mask'].to(device)
            pe = train_data['pe'].to(device)
            y_hat = sentence_classifier(ids, mask, pe)
            y_true = train_data['target'].to(device)
            loss = criterion(y_hat, y_true)
            epoch_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sentence_classifier.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
        epoch_loss = epoch_loss / len(dl_train)
        # evaluation
        optimizer.zero_grad()
        eval_loss, p, r, f1, cm = evaluate(
            sentence_classifier, 
            dl_test, 
            criterion, 
            train_params['positive_label'], 
            device
        )
        #storing metrics
        metrics[epoch] = np.array([epoch_loss, eval_loss, p, r, f1])
        confusion_matrices[epoch] = cm
        end_epoch = time.perf_counter()
        print('finished! Time: ', time.strftime("%Hh%Mm%Ss", time.gmtime(end_epoch - start_epoch)))
    end_train = time.perf_counter()
    
    return metrics, confusion_matrices, time.strftime("%Hh%Mm%Ss", time.gmtime(end_train - start_train))
