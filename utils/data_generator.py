import pandas as pd
import numpy as np
import random
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

class Data_Generator:
    def __init__(self,vocab,batch_size,data_file):
        self.vocab = vocab
        self.batch_size = batch_size
        self.data_file = data_file
        
        self.load_data()
        
    def load_data(self):
        data = pd.read_csv(self.data_file)
        
        questions_1 = data['question1']
        questions_2 = data['question2']
        labels = data['is_duplicate']
        
        self.q1s = []
        self.q2s = []
        self.labels = []
        
        for question1,question2,label in zip(questions_1,questions_2,labels):
            question_1_tokens = text_to_word_sequence(question1)
            question_2_tokens = text_to_word_sequence(question2)
            
            q1 = [self.vocab.get_word_index(word) for word in question_1_tokens]
            q2 = [self.vocab.get_word_index(word) for word in question_2_tokens]
            
            self.q1s.append(q1)
            self.q2s.append(q2)
            
            self.labels.append(label)
    
    def gen_next(self):
        while True:
            X_1 = []
            X_2 = []
            
            Y = []
            
            batch_dpts_indices = np.random.choice(len(self.q1s), self.batch_size)
            for pt_index in range(self.batch_size):
                
                
                X_1.append(self.q1s[batch_dpts_indices[pt_index]])
                X_2.append(self.q2s[batch_dpts_indices[pt_index]])
                
                Y.append(self.labels[batch_dpts_indices[pt_index]])
                
            
            Y = np.expand_dims(np.array(Y),axis=1)
            X_1 = pad_sequences(X_1)
            X_2 = pad_sequences(X_2)
            
            # Randomly switch q1 and q2
            if random.uniform(0, 1) <= 0.5:
                yield [X_1,X_2],Y
            else:
                yield [X_2,X_1],Y
            