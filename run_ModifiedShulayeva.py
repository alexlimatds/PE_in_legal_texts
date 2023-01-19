import application_sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy, scipy, re
import numpy as np

class ModifiedShulayevaEncoder:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm", disable=["ner"])
        
    def preprocess(self, str):
        pstr = str
        pstr = re.sub(r'[/(){}\[\]\|@,;]', ' ', pstr) # replaces symbols with spaces
        pstr = re.sub(r'[^0-9a-z #+_]', '', pstr)     # removes bad symbols
        pstr = re.sub(r'\d+', '', pstr)               # removes numbers
        return pstr
    
    def spacy_tokenizer(self, sentence):
        return [token.text for token in self.nlp(sentence.lower())]
    
    def get_sentences_and_labels(self, dic_docs):
        labels = []
        sentences = []
        positions = []
        for doc_id, df in dic_docs.items():
            s_ind_doc = df['sentence'].to_list()
            sentences.extend(s_ind_doc)
            labels.extend(df['label'].tolist())
            # computing position of each sentence occupys in its source document
            positions.append(
                (np.arange(len(s_ind_doc)) / len(s_ind_doc)).reshape((-1,1))
            )
        positions = np.vstack(positions)
        
        return sentences, labels, positions
    
    def train(self, dic_docs):        
        sentences, _, _ = self.get_sentences_and_labels(dic_docs)
        
        # PoS Tags
        tags = []
        for s in sentences:
            tags.append(
                " ".join([token.tag_ for token in self.nlp(s)])
            )
        self.tfidf_pos_tags = TfidfVectorizer(
            tokenizer=self.spacy_tokenizer, 
            ngram_range=(1, 1), 
            max_features=None
        )
        self.tfidf_pos_tags.fit(tags)
        
        # Dependency pairs
        pairs = []
        for s in sentences:
            pairs_in_s = []
            for token in self.nlp(s):
                if token.dep_ not in ['punct', 'dep'] and token.text != token.head.text:
                    if token.dep_ == 'quantmod':
                        pairs_in_s.append(f'{token.text}-NUMBER')
                    elif token.dep_ == 'nummod':
                        pairs_in_s.append(f'NUMBER-{token.head.text}')
                    else:
                        pairs_in_s.append(f'{token.text}-{token.head.text}')
            pairs.append(' '.join(pairs_in_s))
        self.tfidf_pairs = TfidfVectorizer(
            tokenizer=self.spacy_tokenizer, 
            ngram_range=(1, 1), 
            max_features=2000
        )
        self.tfidf_pairs.fit(pairs)
        
        # TF-IDF
        self.tfidf = TfidfVectorizer(
            tokenizer=self.spacy_tokenizer, 
            preprocessor=self.preprocess, 
            ngram_range=(1, 3), 
            max_features=2000
        )
        self.tfidf.fit(sentences)
        
    def get_features_info(self):
        V_pos = len(self.tfidf_pos_tags.vocabulary_)
        V_pairs = len(self.tfidf_pairs.vocabulary_)
        V_unigrams = len(self.tfidf.vocabulary_)
        embedding_dim = V_pos + V_pairs + V_unigrams + 2 # 1 => sentence lenght, 1 => sentence position
        return f'embedding dim: {embedding_dim} => PoS tag features: {V_pos}, Dependecy pair features: {V_pairs}, N-grams features: {V_unigrams}'

    def merge_features(self, *feature_tensors):
        merged = []
        for tensor in feature_tensors:
            if isinstance(tensor, scipy.sparse.csr_matrix):
                tensor = tensor.toarray()
            merged.append(tensor)
        return np.hstack(merged)
    
    def encode(self, dic_docs):
        sentences, targets, positions = self.get_sentences_and_labels(dic_docs)
        pos_features = self.tfidf_pos_tags.transform(sentences)
        pair_features = self.tfidf_pairs.transform(sentences)
        tfidf_features = self.tfidf.transform(sentences)
        # sentences lengths
        len_features = np.array([len(s) for s in sentences]).reshape((-1,1))
        return self.merge_features(tfidf_features, positions, len_features, pos_features, pair_features), targets


if __name__ == "__main__":
    model_reference = 'ModifiedShulayeva'
    #application_sparse.evaluate_sparse(ModifiedShulayevaEncoder(), model_reference, n_docs=2) # for tests
    application_sparse.evaluate_sparse(ModifiedShulayevaEncoder(), model_reference)
