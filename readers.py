import numpy as np
import re
import itertools
from collections import Counter

from keras.utils.np_utils import to_categorical

# 6 labels + '0' (non tag)
NE = ['O', 'AdverseReaction', 'Animal', 'DrugClass', 'Factor', 'Negation', 'Severity']

class reader(object):
    def __init__(self):
        self.ne = []

    def get_stat(self, lists):
        print len(lists)
        stats = [0] * len(lists[0])
        for lst in lists:
            stats[lst.index(1)] += 1
        print 'stats', stats

    def load_all_data(self):
        """

        :return:
        """

        words, x, y, vocabulary, vocabulary_inv_list = self.load_data("./data/train.txt")
        vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}

        # shuffle_indices = np.random.permutation(np.arange(len(y)))
        # x = x[shuffle_indices]
        # y = y[shuffle_indices]
        # train_len = int(len(x) * 0.9)
        # x_train = x[:train_len]
        # y_train = y[:train_len]
        # x_test = x[train_len:]
        # y_test = y[train_len:]
        # # Shuffle data
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_train = x[shuffle_indices]
        y_train = y[shuffle_indices]
        # train_len = int(len(x) * 0.9)
        self.get_stat(y_train.tolist())


        words, x, y, vocabulary, vocabulary_inv_list = self.load_data("./data/test.txt")
        # vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}

        # Shuffle data
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_test = x[shuffle_indices]
        y_test = y[shuffle_indices]

        self.get_stat(y_test.tolist())

        return words, x_train, y_train, x_test, y_test, vocabulary_inv

    def load_data(self, datadir):
        """
        Loads and preprocessed data for the MR dataset.
        Returns input vectors, labels, vocabulary, and inverse vocabulary.
        """
        # Load and preprocess data
        print datadir
        words, labels = self.load_data_and_labels(datadir)
        vocabulary, vocabulary_inv = self.build_vocab(words)

        x = np.array([[vocabulary[word]] for word in words])
        y = np.array(labels)

        return [words, x, y, vocabulary, vocabulary_inv]


    def load_data_and_labels(self, datadir):
        """
        Loads MR polarity data from files, splits the data into words and generates labels.
        Returns split sentences and labels.
        """
        # Load data from files
        data = list(open(datadir).readlines())
        data = [s.strip() for s in data]
        x_text = []
        labels = []
        stastics = [0]* len(NE)
        for line in data:
            if not line.startswith('-DOCSTART-') and len(line) != 0:
                info = line.split(' ')
                word = info[0]
                gram = info[1]
                label = info[2]
                if '-' in label:
                    label = label.split('-')[1]
                # if label != 'O':
                x_text.append(word)
                labels.append(label)
                if label not in self.ne:
                    self.ne.append(label)
        print sorted(self.ne)
        for i in range(len(labels)):
            tmp = [0] * len(NE)
            # if labels[i] != 'O':
            index = NE.index(labels[i])
            tmp[index] = 1
            labels[i] = tmp
            stastics[index] += 1

                # print labels[i]

            # labels[i] = NE.index(labels[i])

        # print(len(x_text))
        # print(len(labels))

        # labels = self.to_one_hot(labels)
        # print labels
        y = np.asarray(labels)

        print stastics
        # print(len(y))
        # print(len(positive_labels) + len(negative_labels))
        # print(len(negative_labels))
        return [x_text, y]


    def build_vocab(self, words):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # Build vocabulary
        word_counts = Counter(words)
        print word_counts
        # Mapping from index to word
        # for x in word_counts.most_common():
        #     print x
        vocabulary_inv = [x[0] for x in word_counts.most_common()]
        print 'words', words[:50]
        print 'inventory', vocabulary_inv[:50]
        # Mapping from word to index
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        return [vocabulary, vocabulary_inv]


if __name__ == '__main__':
    rd = reader()
    x_train, y_train, x_test, y_test, vocabulary_inv = rd.load_all_data()
    # rd.build_vocab(x_train)