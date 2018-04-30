from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
import pickle

class Vocabulary:
    
    def setup_corpus_vocabulary(self,data_file,max_words):
        data = pd.read_csv(data_file)
        
        all_questions = []

        all_question_1 = data['question1']
        all_question_2 = data['question2']

        for question in all_question_1:
            all_questions.append(question)

        for question in all_question_2:
            all_questions.append(question)
            
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(all_questions)
        
        self.word_index = tokenizer.word_index
        
        # Increment the index by 1 to take padding token at index 0
        self.word_index = {word:index+1 for word,index in self.word_index.items()}
        
        # Add padding and unkown token
        self.word_index['<PAD>'] = 0
        # self.word_index['<UNK>'] = len(self.word_index)
        
        self.index_word = {index:word for word,index in self.word_index.items()}
        
    def load_glove_vocabulary(self,glove_file):
        
        self.word_embeddings = {}
        with open(glove_file, encoding='utf-8') as f:
            for line in f:
                values = line.split(' ')
                word = values[0]
                embedding = np.asarray(values[1:], dtype='float32')
                self.word_embeddings[word] = embedding
                self.embedding_dim = embedding.shape[0]
    
    def construct_embedding_matrix(self):
        
        self.word_embedding_matrix = np.zeros((len(self.word_index) + 1, self.embedding_dim))
        for word, i in self.word_index.items():
            embedding_vector = self.word_embeddings.get(word)
            if embedding_vector is not None:
                self.word_embedding_matrix[i] = embedding_vector
            
    def get_word_index(self,word):
        
        if word in self.word_index:
            return self.word_index[word]
        else:
            return len(self.word_index)
        
    def load_word_index(self,pickle_path):
        with open(pickle_path, 'rb') as handle:
            self.word_index = pickle.load(handle)
            
    def save_word_index(self,pickle_path):
        with open(pickle_path, 'wb') as handle:
            pickle.dump(self.word_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    
        
        