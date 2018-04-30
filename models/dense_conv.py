import numpy as np
import keras
import sys
sys.path.append('/home/jupyter/notebooks/src')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf

from keras.models import Model as K_Model
from keras.layers import Input, Bidirectional, LSTM, dot, Flatten, Dense, Reshape, add, Dropout, BatchNormalization,Concatenate,Merge,Lambda, TimeDistributed, Conv1D, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint
from keras import backend as K

class Model:
    
    def __init__(self,vocab):
        self.vocab = vocab
        self.build_model()
        
    def build_model(self):
        
        question1 = Input(shape=(None,))
        question2 = Input(shape=(None,))

        embedder = Embedding(self.vocab.word_embedding_matrix.shape[0], self.vocab.word_embedding_matrix.shape[1], embeddings_initializer=keras.initializers.constant(self.vocab.word_embedding_matrix), trainable=False)
        
        q1_encoded = embedder(question1)
        q2_encoded = embedder(question2)
        
        # Build Dense Stream
        q1_dense = TimeDistributed(Dense(300))(embedder(question1))
        q2_dense = TimeDistributed(Dense(300))(embedder(question2))
        
        q1_dense = Lambda(lambda x: K.max(x,axis=1),output_shape=lambda x: (x[0],x[2]))(q1_dense)
        q2_dense = Lambda(lambda x: K.max(x,axis=1),output_shape=lambda x: (x[0],x[2]))(q2_dense)

        
        # Build Conv stream
        q1_conv_1 = Conv1D(64,5,padding='valid',activation='relu')(q1_encoded)
        q1_conv_1 = Dropout(0.2)(q1_conv_1)
        
        q1_conv_2 = Conv1D(64,5,padding='valid',activation='relu')(q1_conv_1)
        q1_conv_2 = Dropout(0.2)(q1_conv_2)
        
        q1_conv_out = GlobalMaxPooling1D()(q1_conv_2)
        q1_conv_out = Dropout(0.2)(q1_conv_out)

        q1_conv_out = Dense(300)(q1_conv_out)
        q1_conv_out = Dropout(0.2)(q1_conv_out)
        
        q2_conv_1 = Conv1D(64,5,padding='valid',activation='relu')(q2_encoded)
        q2_conv_1 = Dropout(0.2)(q2_conv_1)
        
        q2_conv_2 = Conv1D(64,5,padding='valid',activation='relu')(q2_conv_1)
        q2_conv_2 = Dropout(0.2)(q2_conv_2)
        
        q2_conv_out = GlobalMaxPooling1D()(q2_conv_2)
        q2_conv_out = Dropout(0.2)(q2_conv_out)

        q2_conv_out = Dense(300)(q2_conv_out)
        q2_conv_out = Dropout(0.2)(q2_conv_out)
        
        merged = Merge(mode=lambda x: tf.stack([x[0],x[1],x[2],x[3]],axis=1), output_shape=lambda x: (x[0][0],1200))([q1_dense,q2_dense,q1_conv_out,q2_conv_out])
        
        fc_1 = Dense(300,activation='relu')(merged)
        fc_1 = Dropout(0.1)(fc_1)
        
        fc_2 = BatchNormalization()(fc_1)
        fc_2 = Dense(200,activation='relu')(fc_2)
        fc_2 = Dropout(0.1)(fc_2)
        
        fc_3 = BatchNormalization()(fc_2)
        fc_3 = Dense(64,activation='relu')(fc_3)
        fc_3 = Dropout(0.1)(fc_3)
        
        fc_4 = BatchNormalization()(fc_3)
        fc_4 = Dense(16,activation='relu')(fc_4)
        fc_4 = Dropout(0.1)(fc_4)
        
        output = BatchNormalization()(fc_4)
        output = Dense(1,activation='sigmoid')(fc_4)
        
        # Pack it all up into a model
        self.model = K_Model([question1, question2], output)
        self.model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.007), metrics=['accuracy'])
        