import application_sparse
import re
from sklearn.feature_extraction.text import TfidfVectorizer

class TfIdfEncoder:
    def preprocess(self, str):
        pstr = str
        pstr = re.sub(r'[/(){}\[\]\|@,;]', ' ', pstr) # replaces symbols with spaces
        pstr = re.sub(r'[^0-9a-z #+_]', '', pstr)     # removes bad symbols
        pstr = re.sub(r'\d+', '', pstr)               # removes numbers
        return pstr
    
    def get_sentences_and_labels(self, dic_docs):
        labels = []
        sentences = []
        for doc_id, df in dic_docs.items():
            sentences.extend(df['sentence'].to_list())
            labels.extend(df['label'].tolist())

        return sentences, labels
    
    def train(self, dic_docs):
        sentences, _ = self.get_sentences_and_labels(dic_docs)
        self.model = TfidfVectorizer(
            preprocessor=self.preprocess, 
            ngram_range=(1, 3), 
            max_features=2000
        )
        self.model.fit(sentences)
        
    def get_features_info(self):
        return f'embedding dim: {len(self.model.vocabulary_)}'
        
    def encode(self, dic_docs):
        sentences, targets = self.get_sentences_and_labels(dic_docs)
        return self.model.transform(sentences).toarray(), targets
        

if __name__ == "__main__":
    model_reference = 'TF-IDF'
    application_sparse.evaluate_sparse(TfIdfEncoder(), model_reference)
