import numpy as np
from readers import reader
import readers
import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Activation, Flatten, Conv1D, MaxPooling1D
from keras.preprocessing import sequence

np.random.seed(0)
logging.basicConfig(format='%(asctime)s: %(message)s',
                    level=logging.INFO,
                    datefmt='%m/%d/%Y %I:%M:%S %p')
verbose = 1

undersampling = 5
embedding_dim = 50

# Training parameters
batch_size = 64
num_epochs = 5

# Prepossessing parameters
sequence_length = 300
final_vec_size = len(readers.NE)


def under_sample(n, x_train, x_test, y_train, vocabulary_inv):
    models = []
    print 'under sample --------------------------------------------------'
    # print x_train
    # print y_train

    labels_0 = []
    labels_non_0 = []

    words_0 = []
    words_non_0 = []

    for i in range(len(y_train)):
        if y_train[i][0] == 1:
            labels_0.append(y_train[i])
            words_0.append(x_train[i])
        else:
            labels_non_0.append(y_train[i])
            words_non_0.append(x_train[i])

    print np.asarray(labels_0)
    print np.asarray(labels_non_0)

    new_x_train = []
    new_y_train = []

    w = int(len(words_0) / n)

    for i in range(n):
        print np.asarray(words_0[i * w: (i + 1) * w]).shape
        print np.asarray(words_non_0).shape
        x = np.concatenate((words_0[i * w: (i + 1) * w], words_non_0), axis=0)
        new_x_train.append(x)

        y = np.concatenate((labels_0[i * w: (i + 1) * w], labels_non_0), axis=0)
        new_y_train.append(y)

    for i in range(n):
        model = train_model(new_x_train[i], x_test, new_y_train[i], vocabulary_inv)
        models.append(model)

    return models


def train_model(x_train, x_test, y_train, vocabulary_inv):
    model = None

    if verbose > 0:
        print("x_train shape:", x_train.shape)
        print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

    # Build model
    input_shape = (sequence_length,)

    model = Sequential()
    model.add(Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length))
    model.add(Conv1D(32, kernel_size=3,
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, 8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(final_vec_size, activation='sigmoid'))
    # model.add(Activation('softmax'))
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])


    # Train the model
    if verbose > 0:
        print 'x_train', len(x_train), x_train
        print 'y_train', len(y_train), y_train
    model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs)

    return model


def print_confusion_matrix(c_matrix):
    tmp = '\nGold \ Machine'.ljust(18)
    for a in list(c_matrix.get(0)):
        tmp += str(a).ljust(7)

    tmp += '\n'
    for key, elem in c_matrix.items():
        tmp += str(key).ljust(17)
        for k, e in elem.items():
            tmp += str(e).ljust(7)
        tmp += '\n'

    print tmp


def main():
    # Data Preparation
    logging.info("Load data...")
    read = reader()
    test_words, x_train, y_train, x_test, y_test, vocabulary_inv = read.load_all_data()
    print 'before padding', len(x_train)
    x_test = sequence.pad_sequences(x_test, maxlen=sequence_length)
    x_train = sequence.pad_sequences(x_train, maxlen=sequence_length)

    if verbose > 0:
        print 'x_train'
        print x_train
        print 'y_train'
        print y_train

    models = under_sample(undersampling, x_train, x_test, y_train, vocabulary_inv)

    prds = []
    predictions = []
    stats_gold = [0] * len(readers.NE)
    stats_output = [0] * len(readers.NE)

    for model in models:
        predictions = model.predict(x_test)
        for i in range(len(predictions)):
            # if np.argmax(y_test[i]) != 0:
            stats_gold[list(y_test[i]).index(1)] += 1
            stats_output[int(np.argmax(predictions[i]))] += 1
        print 'gold', stats_gold
        print 'output', stats_output

        if len(prds) == 0:
            prds = predictions
        else:
            print 'prediction adding up'
            for i in range(len(predictions)):
                prds[i] = np.add(prds[i], predictions[i])
    print predictions

    predictions = prds
    accuracy = 0
    stats_gold = [0] * len(readers.NE)
    stats_output = [0] * len(readers.NE)

    # initialize confusion matrix
    sub_matrix = {}
    for i in range(len(readers.NE)):
        sub_matrix.update({i: 0})

    c_matrix = {}
    for i in range(len(readers.NE)):
        c_matrix.update({i: sub_matrix})

    with open('result.txt', 'w+') as f:
        for i in range(len(predictions)):
            # if np.argmax(y_test[i]) != 0:
            stats_gold[list(y_test[i]).index(1)] += 1
            stats_output[int(np.argmax(predictions[i]))] += 1

            g = list(y_test[i]).index(1)
            p = int(np.argmax(predictions[i]))
            row = c_matrix.get(g)
            row = row.copy()
            row.update({p: row.get(p) + 1})
            c_matrix.update({g: row})

            # print np.argmax(predictions[i]) == np.argmax(y_test[i]), \
            #     'gold: ', \
            #     np.argmax(y_test[i]), \
            #     'output: ', \
            #     np.argmax(predictions[i]),  \
            #     'all prob: ', \
            #     predictions[i]
            if np.argmax(predictions[i]) == np.argmax(y_test[i]):
                accuracy += 1

            line = test_words[i] + ' NN ' + readers.NE[p] + ' ' + readers.NE[p] + '\n'
            f.write(line)

    print_confusion_matrix(c_matrix)
    print accuracy
    accuracy = accuracy / len(predictions)
    print accuracy
    print 'gold', stats_gold
    print 'output', stats_output


if __name__ == '__main__':
    main()
