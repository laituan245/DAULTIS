import io
import json
import random
import numpy as np
from os.path import join
from nltk import word_tokenize
from sklearn.preprocessing import MinMaxScaler

def get_data(task_name, shuffle_train=True, load_glove=True):
    base_dir = join('data/', task_name)
    glove_fp = 'data/glove.840B.300d.txt'

    # Read files
    image_features=np.load(join(base_dir, 'image_features.npy'), encoding='latin1').item()
    train_data = read_json(join(base_dir, 'train_data.json'))
    test_data = read_json(join(base_dir, 'test_data.json'))
    vocab = set(read_lines(join(base_dir, 'vocab.txt')))
    dictionary = Dictionary(join(base_dir, 'vocab.txt'))
    weights = np.random.rand(dictionary.n_words, 300)
    if load_glove:
        weights = load_glove_vectors(glove_fp, dictionary)

    # Process Train Part
    train_images, train_texts = [], []
    for image, _, captions in train_data:
        train_images.append(image_features[image])
        text = []
        for caption in captions:
            for word in word_tokenize(caption.lower()):
                if word in dictionary.word2index:
                    text.append(dictionary.word2index[word])
        train_texts.append(bow_encoding(text, weights))
    train_images = np.asarray(train_images)
    train_texts = np.asarray(train_texts)

    # Process Test Part
    test_images, test_texts = [], []
    for image, _, captions in test_data:
        test_images.append(image_features[image])
        text = []
        for caption in captions:
            for word in word_tokenize(caption.lower()):
                if word in dictionary.word2index:
                    text.append(dictionary.word2index[word])
        test_texts.append(bow_encoding(text, weights))
    test_images = np.asarray(test_images)
    test_texts = np.asarray(test_texts)

    # Image Feature Scaling
    images_scaler = MinMaxScaler()
    images_scaler.fit(train_images)
    train_images = images_scaler.transform(train_images)
    test_images = images_scaler.transform(test_images)
    train_images = list(train_images)
    test_images = list(test_images)

    # Text Feature Scaling
    text_scaler = MinMaxScaler()
    text_scaler.fit(train_texts)
    train_texts = text_scaler.transform(train_texts)
    test_texts = text_scaler.transform(test_texts)
    train_texts = list(train_texts)
    test_texts = list(test_texts)

    # Shuffle Traing Data
    if shuffle_train:
        random.shuffle(train_images)
        random.shuffle(train_texts)

    return AugmentedList(train_images), AugmentedList(train_texts), \
           AugmentedList(test_images), AugmentedList(test_texts)

# Helper classes and functions
class Dictionary:
    PAD_token = 0
    UNK_token = 1

    def __init__(self, fn):
        # Intialization
        self.word2index = {}
        self.index2word = {0: "[PAD]", 1: "[UNK]"}
        self.n_words = 2

        # Load words from file
        f = io.open(fn, encoding='utf-8')
        for word in f:
            self.add_word(word.strip())
        f.close()

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def indexes_to_words(self, indexes):
        index2word = self.index2word
        return [index2word[index] for index in indexes]

    def words_to_indexes(self, words, max_len = None):
        word2index = self.word2index
        PAD_token = self.PAD_token
        UNK_token = self.UNK_token
        indexes = []
        for word in words:
            if len(word) > 0:
                if word in word2index:
                    indexes.append(word2index[word])
                else:
                    indexes.append(UNK_token)
        if max_len:
            while len(indexes) < max_len:
                indexes.append(PAD_token)
            indexes = indexes[:max_len]
        return indexes

class AugmentedList:
    def __init__(self, items, shuffle_between_epoch=False):
        self.items = items
        self.cur_idx = 0
        self.shuffle_between_epoch = shuffle_between_epoch

    def next_items(self, batch_size):
        items = self.items
        start_idx = self.cur_idx
        end_idx = start_idx + batch_size
        if end_idx <= self.size:
            self.cur_idx = end_idx % self.size
            return items[start_idx : end_idx]
        else:
            first_part = items[start_idx : self.size]
            remain_size = batch_size - (self.size - start_idx)
            second_part = items[0 : remain_size]
            self.cur_idx = remain_size
            returned_batch = [item for item in first_part + second_part]
            if self.shuffle_between_epoch:
                random.shuffle(self.items)
            return returned_batch

    @property
    def size(self):
        return len(self.items)

def bow_encoding(indexes, weights):
    encoding = 0.0
    for index in indexes:
        encoding += weights[index, :]
    encoding /= len(indexes)
    return encoding

def load_glove_vectors(path, dictionary, emb_size = 300):
    vocab_size = dictionary.n_words
    word2index = dictionary.word2index
    weights = np.zeros((vocab_size, emb_size))

    with io.open(path, encoding='utf-8') as f:
        for line in f:
            word, emb = line.strip().split(' ', 1)
            if word in word2index:
                weights[word2index[word]] = np.asarray(list(map(float, emb.split(' ')))[:emb_size])
    return weights

def read_json(fn):
    with open(fn) as f:
        data = json.load(f)
    return data

def read_lines(fn):
    lines = []
    f = open(fn, 'r')
    for line in f:
        lines.append(line.strip())
    f.close()
    return lines
