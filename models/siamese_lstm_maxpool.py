import numpy as np
import keras
import sys
sys.path.append('/home/jupyter/notebooks/src')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf

from keras.models import Model as K_Model
from keras.layers import Input, Bidirectional, LSTM, dot, Flatten, Dense, Reshape, add, Dropout, BatchNormalization,Concatenate,Merge,Lambda
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint
from keras import backend as K

class Model:
    
    def __init__(self,vocab):
        self.vocab = vocab
        self.build_model()
        
    def squared_distance(self,left,right):
        return K.sqrt(K.sum(K.square(K.abs(left-right)),axis=1,keepdims=True))
    
    def cosine_distance(self,left,right):
        
        left = K.l2_normalize(left, axis=-1)
        right = K.l2_normalize(right, axis=-1)
        return 1 - K.sum((left * right), axis=1,keepdims=True)

    def cosine_dist_output_shape(input_shape):
        
        return (input_shape[0], 1)
    
    def build_model(self):
        
        question1 = Input(shape=(None,))
        question2 = Input(shape=(None,))

        embedder = Embedding(self.vocab.word_embedding_matrix.shape[0], self.vocab.word_embedding_matrix.shape[1], embeddings_initializer=keras.initializers.constant(self.vocab.word_embedding_matrix), trainable=False)
        
        q1 = embedder(question1)
        q2 = embedder(question2)
        
        lstm = Bidirectional(LSTM(128, return_sequences=True))
        
        left_output = lstm(q1)
        right_output = lstm(q2)
        
        left_output = Lambda(lambda x: K.max(x,axis=1),output_shape=lambda x: (x[0],x[2]))(left_output)
        right_output = Lambda(lambda x: K.max(x,axis=1),output_shape=lambda x: (x[0],x[2]))(right_output)

        merged = Merge(mode=lambda x: tf.stack([x[0],x[1]],axis=1), output_shape=lambda x: (x[0][0],x[0][1] * 2))([left_output,right_output])
        
        fc_1 = Dense(128,activation='relu')(merged)
        fc_1 = BatchNormalization()(fc_1)
        fc_1 = Dropout(0.1)(fc_1)
        fc_2 = Dense(64,activation='relu')(fc_1)
        fc_2 = BatchNormalization()(fc_2)
        fc_2 = Dropout(0.1)(fc_2)
        fc_3 = Dense(16,activation='relu')(fc_2)
        fc_3 = BatchNormalization()(fc_3)
        fc_3 = Dropout(0.1)(fc_3)
        output = Dense(1,activation='sigmoid')(fc_3)
        
        # Pack it all up into a model
        self.model = K_Model([question1, question2], output)
        self.model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.007), metrics=['accuracy'])
        