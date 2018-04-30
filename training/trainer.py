import sys
sys.path.append('/home/jupyter/notebooks/src')

from utils.data_generator import Data_Generator
from utils.vocabulary import Vocabulary
from models.dense_conv import Model
import keras
import os

# Data Paths
train_file = '/home/jupyter/data/train.csv'
test_file = '/home/jupyter/data/test.csv'
complete_file = '/home/jupyter/data/cleaned.csv'
glove_file = '/home/jupyter/data/vocab/glove.6B.100d.txt'
max_nb_words = 200000
exp_name = 'run103'
model_dir = os.path.join('/home/jupyter/data/model_snapshots',exp_name)
log_dir = os.path.join('/home/jupyter/tf-logs/',exp_name)

os.makedirs(model_dir,exist_ok=True)
os.makedirs(log_dir,exist_ok=True)

vocab = Vocabulary()
vocab.setup_corpus_vocabulary(complete_file,max_nb_words)
print('Setup Vocab')
vocab.load_glove_vocabulary(glove_file)
print('Loaded glove')
vocab.construct_embedding_matrix()
print('Loaded embedding matrix')

arch = Model(vocab)
print(arch.model.summary())

train_gen = Data_Generator(vocab,16,train_file)
train_gen.load_data()
test_gen = Data_Generator(vocab,16,test_file)
test_gen.load_data()

file_name = 'dense-conv'
check_cb = keras.callbacks.ModelCheckpoint(os.path.join(model_dir,file_name + '.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5'),
                                   monitor='val_acc',
                                   verbose=0, save_best_only=True, mode='max')
tbCallback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=1, mode='auto')

arch.model.fit_generator(generator = train_gen.gen_next(), steps_per_epoch=20000, epochs=20, verbose=1
                         , callbacks=[check_cb,tbCallback]
                         , validation_data=test_gen.gen_next(), validation_steps=500, max_queue_size=10
                         , workers=3, use_multiprocessing=True, shuffle=True, initial_epoch=0)