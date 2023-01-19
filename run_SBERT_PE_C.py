import application_sbert, functions
import torch

def get_features(dic_docs, sent_encoder):
    """
    Generates the features for a set of documents.
    Arguments:
        dic_docs : a dictionary as returned by the read_docs function.
        sent_encoder : SBERT instance.
    Returns:
        - The features as a numpy matrix. Each line in the matrix is respective to a sentence.
        - A list of strings with respective sentences' labels.
        - A dictionary associating each document with its features. The key is the 
        document ID and the value is a numpy matrix.
    """
    features = None
    labels = []
    features_by_doc = {}
    
    # search for the maximum number of sentences in all documents to compute PE matrix just one time
    max_seq_len = 0
    for doc_id, df in dic_docs.items():
        max_seq_len = max(max_seq_len, df.shape[0])
    
    # PE matrix
    embedding = sent_encoder.encode('dummy text', convert_to_tensor=True)
    PE = functions.getPositionEncoding(max_seq_len, embedding.shape[0]).to(embedding.device)
    
    for doc_id, df in dic_docs.items():
        sentences = df['sentence'].to_list()
        embedding = sent_encoder.encode(sentences, convert_to_tensor=True)
        embedding = torch.hstack( # concatenation of SBERT features and PE features
            (
                embedding, 
                PE[0:len(sentences), :]
            )
        )
        features_by_doc[doc_id] = embedding
        if features is None:
            features = embedding
        else:
            features = torch.vstack((features, embedding))
        labels.extend(df['label'].tolist())
    for doc_id, tensor in features_by_doc.items():
        features_by_doc[doc_id] = tensor.detach().to('cpu').numpy()
    return features.detach().to('cpu').numpy(), labels, features_by_doc


if __name__ == "__main__":
    model_reference = 'SBERT_PE_C'
    application_sbert.evaluate_sbert(get_features, model_reference)
