import pickle
import keras
from keras.layers import Embedding, Bidirectional, LSTM
from keras.layers import Activation, Dense
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras_contrib.layers import CRF
import keras_contrib

EMBED_DIM = 200
BiRNN_UNITS = 200

def load_data():
    source_data = 'data/data-small.pkl'
    with open(source_data, 'rb') as f:
        data_x = pickle.load(f)
        data_y = pickle.load(f)
        word2id = pickle.load(f)
        id2word = pickle.load(f)
        tag2id = pickle.load(f)
        id2tag = pickle.load(f)

        return data_x, data_y, word2id, id2word, tag2id, id2tag

def create_model(train=True):
    if train:
        data_x, data_y, word2id, id2word, tag2id, id2tag = load_data()

        # print("data_x:"),print(data_x)
        # print("tag2id:"),print(tag2id)
        data_y = data_y.reshape((data_y.shape[0], data_y.shape[1], 1))
        # print("__data_y:"),print(data_y)
        train_x1, test_x, train_y1, test_y = train_test_split(data_x, data_y, test_size=0.3, random_state=40)

        train_x = data_x
        train_y = data_y

        vocab = word2id.keys()
        chunk_tags = tag2id.keys()

        # print("vocab:"),print(vocab)
        # print('chunk_tags:'),print(chunk_tags)

    else:
        data_x, data_y, word2id, id2word, tag2id, id2tag = load_data()
        vocab = word2id.keys()
        chunk_tags = tag2id.keys()
    model = Sequential()
    model.add(Embedding(len(vocab) + 1, EMBED_DIM, mask_zero=True))  # Random embedding
    model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
    model.add(Dense(20, activation='softmax'))
    model.summary()
    model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model = Sequential()
    # model.add(Embedding(len(vocab), EMBED_DIM, mask_zero=True))  # Random embedding
    # model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
    # crf = CRF(len(chunk_tags), sparse_target=True)
    # model.add(crf)
    # model.summary()
    # model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    if train:
        return model, (train_x, train_y), (test_x, test_y)
    else:
        return model, (vocab, chunk_tags)
