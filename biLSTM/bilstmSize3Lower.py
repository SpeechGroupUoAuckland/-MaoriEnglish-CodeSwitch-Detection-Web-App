import pickle
import gc
import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
# from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import pad_sequences
from keras.models import Sequential
from keras.layers import Bidirectional, Dense, Dropout, Embedding, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

def vectorize_notes(col, MAX_NB_WORDS, verbose=True):
    """Takes a note column and encodes it into a series of integer
        Also returns the dictionnary mapping the word to the integer"""
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(col)
    data = tokenizer.texts_to_sequences(col)
    note_length = [len(x) for x in data]
    vocab = tokenizer.word_index
    MAX_VOCAB = len(vocab)
    if verbose:
        print('Vocabulary size: %s' % MAX_VOCAB)
        print('Average note length: %s' % np.mean(note_length))
        print('Max note length: %s' % np.max(note_length))
    return data, vocab, MAX_VOCAB, tokenizer


def pad_notes(data, MAX_SEQ_LENGTH):
    data = pad_sequences(data, maxlen=MAX_SEQ_LENGTH)
    return data, data.shape[1]


def embedding_matrix(f_name, dictionary, EMBEDDING_DIM, verbose=True, sigma=None):
    """Takes a pre-trained embedding and adapts it to the dictionary at hand
        Words not found will be all-zeros in the matrix"""

    # Dictionary of words from the pre trained embedding
    pretrained_dict = {}
    with open(f_name, 'rb') as f:
        for line in f:
            try:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:])
                pretrained_dict[word] = coefs
            except ValueError:
                continue

    if sigma:
        pretrained_matrix = sigma * \
            np.random.rand(len(dictionary) + 1, EMBEDDING_DIM)
    else:
        pretrained_matrix = np.zeros((len(dictionary) + 1, EMBEDDING_DIM))

    # Substitution of default values by pretrained values when applicable
    for word, i in dictionary.items():
        try:
            vector = pretrained_dict.get(word)
            if vector is not None:
                pretrained_matrix[i] = vector
        except ValueError:
            continue

    if verbose:
        print('Vocabulary in notes:', len(dictionary))
        print('Vocabulary in original embedding:', len(pretrained_dict))
        inter = list(set(dictionary.keys()) & set(pretrained_dict.keys()))
        print('Vocabulary intersection:', len(inter))

    return pretrained_matrix, pretrained_dict


def train_val_test_split(X, y, val_size=0.1, test_size=0.2, random_state=101):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(test_size), random_state=random_state)
    return X_train, X_test, y_train, y_test


def main():
    WINDOW_SIZE = 3

    # Read Hansard csv file
    df = pd.read_csv(f'../20220321_Hansard_DB_MP_only_window_{WINDOW_SIZE}.csv')[['text', 'Labels_Final']]
    df.columns = ['TEXT', 'label']
    df.drop(df[(df.label == 'N') | (df.label == 'U')].index, inplace=True)
    df['TEXT'] = df['TEXT'].astype(str).str.lower()

    # preprocess notes
    MAX_VOCAB = None
    MAX_SEQ_LENGTH = WINDOW_SIZE
    text = df.TEXT
    data_vectorized, dictionary, MAX_VOCAB, tokenizer = vectorize_notes(
        text, MAX_VOCAB, verbose=True)
    data, MAX_SEQ_LENGTH = pad_notes(data_vectorized, MAX_SEQ_LENGTH)

    print("Final Vocabulary: %s" % MAX_VOCAB)
    print("Final Max Sequence Length: %s" % MAX_SEQ_LENGTH)

    EMBEDDING_DIM = 300
    EMBEDDING_MATRIX = []

    # Embedding matrix
    EMBEDDING_LOC = 'miwiki_model2_MPW300SG.vec'
    EMBEDDING_MATRIX, embedding_dict = embedding_matrix(
        EMBEDDING_LOC, dictionary, EMBEDDING_DIM, verbose=True, sigma=True)

    X = data
    Y = pd.get_dummies(df['label']).values
    print('Shape of label tensor:', Y.shape)

    # Split sets
    X_train, X_test, y_train, y_test = train_val_test_split(
        X, Y, val_size=0.2, test_size=0.1, random_state=101)
    print("Train: ", X_train.shape, y_train.shape)
    print("Test: ", X_test.shape, y_test.shape)

    # Callbacks
    callback = EarlyStopping(monitor='loss', patience=3)

    # Checkpoint
    checkpoint_path = "../models/biLSTM/lowerCase/checkpoint"
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_weight_only=True,
                                    save_best_only=False, verbose=1, mode='min')
    
    # Build model
    model = Sequential()
    embedding = Embedding(MAX_VOCAB + 1, EMBEDDING_DIM,
                          weights=[EMBEDDING_MATRIX], input_length=MAX_SEQ_LENGTH, trainable=False)

    model.add(embedding)
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    model.summary()

    # Release memory
    del data, X,  Y, df
    gc.collect()

    # Train model
    history = model.fit(X_train, y_train, callbacks=[callback, checkpoint],
                        batch_size=64, epochs=150,
                        validation_split=0.1,
                        verbose=2)

    # Save model
    model.save(f'../models/biLSTM/lowerCase/bilstmSize{WINDOW_SIZE}.h5')

    # Save Tokenizer i.e. Vocabulary
    with open(f'../models/biLSTM/lowerCase/tokenizerBilstmSize{WINDOW_SIZE}.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Evaluate model
    y_out = model.predict(X_test, batch_size=64)
    y_pred = np.where(y_out > 0.5, 1, 0)

    print(classification_report(y_test, y_pred))

    with open(f'../models/biLSTM/lowerCase/f1ScoreMulticlassBilstmHansardSize{WINDOW_SIZE}.txt', 'w') as f:
        print('classification report', classification_report(
            y_test, y_pred, digits=4), file=f)
        print(model.summary(), file=f)


if __name__ == '__main__':
    main()
