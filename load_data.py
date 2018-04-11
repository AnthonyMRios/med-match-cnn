import re
import random
import json
import numpy as np
import gensim
from gensim.models.keyedvectors import KeyedVectors
import nltk
nltk.download('punkt')

def load_data_file(txt_filename):
    txt = open(txt_filename, 'r')
    X_txt = []
    Y = []
    for row in txt:
        data = json.loads(row.strip())
        #X_txt.append(' '.join(nltk.word_tokenize(data['text'])))
        if 'txt' in data:
            X_txt.append(data['txt'])
        else:
            X_txt.append(data['text'])
        Y.append([x for x in data['labels'] if x != ''])
    txt.close()
    return X_txt, Y

class ProcessData(object):
    def __init__(self, pretrain_wv=None, lower=True, min_df=5):
        self.pattern = re.compile(r'(?u)\b\w\w+\b')
        #self.pattern = re.compile('[A-Z][a-z]+')
        self.min_df = min_df
        self.lower = lower
        if pretrain_wv is not None:
            #self.wv = gensim.models.Word2Vec.load(pretrain_wv)
            self.wv = KeyedVectors.load_word2vec_format('/home/amri228/chemprot/data2/glove/glove_300d_w2v_format.txt', binary=False)
        else:
            self.wv = None
        self.embs = [np.zeros((300,)),
            np.random.random((300,))*0.01]
        self.word_index = {None:0, 'UNK':1}

    def _tokenize(self, string):
        if self.lower:
            example = string.strip().lower()
        else:
            example = string.strip().lower()
        #return nltk.word_tokenize(string)
        return re.findall(self.pattern, example)

    def fit(self, data):
        token_cnts = {}
        for ex in data:
            example_tokens = self._tokenize(ex)
            for token in example_tokens:
                if token not in token_cnts:
                    token_cnts[token] = 1
                else:
                    token_cnts[token] += 1

        index = 2
        for value, key in enumerate(token_cnts):
            if value < self.min_df:
                continue
            self.word_index[key] = index
            if self.wv is not None:
                if key in self.wv:
                    self.embs.append(self.wv[key])
                else:
                    self.embs.append(np.random.uniform(-1., 1., (300,)))
                    #self.embs.append(np.random.random((300,))*0.01)
            else:
                self.embs.append(np.random.uniform(-1., 1., (300,)))
                #self.embs.append(np.random.random((300,))*0.01)
            index += 1

        self.embs = np.array(self.embs)
        del self.wv
        return

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def transform(self, data):
        return_dataset = []
        for ex in data:
            example = self._tokenize(ex)
            index_example = []
            for token in example:
                if token in self.word_index:
                    index_example.append(self.word_index[token])
                else:
                    index_example.append(self.word_index['UNK'])
            return_dataset.append(index_example)

        return return_dataset

    def pad_data(self, data, to_shuffle=False):
        max_len = np.max([len(x) for x in data])
        padded_dataset = []
        for ex in data:
            if to_shuffle:
                #example = random.sample(ex, len(ex))
                example = ex
            else:
                example = ex
            zeros = [0]*(max_len-len(example))
            padded_dataset.append(example+zeros)
        return np.array(padded_dataset)

    def pad_data_hier(self, data):
        max_sents = np.max([len(x) for x in data])
        max_len = np.max([len(x) for y in data for x in y])
        padded_dataset = []
        for par in data:
            pad_sents = []
            for example in par:
                zeros = [0]*(max_len-len(example))
                pad_sents.append(example+zeros)
            for x in range(max_sents-len(par)):
                zeros = [0]*max_len
                pad_sents.append(zeros)
            padded_dataset.append(pad_sents)
        return np.array(padded_dataset)

class ProcessHierData(object):
    def __init__(self, pretrain_wv=None, lower=True, min_df=5):
        self.pattern = re.compile(r'(?u)\b\w\w+\b')
        self.min_df = min_df
        self.lower = lower
        if pretrain_wv is not None:
            self.wv = gensim.models.Word2Vec.load(pretrain_wv)
        else:
            self.wv = None
        self.embs = [np.zeros((300,)),
            np.random.random((300,))*0.01]
        self.word_index = {None:0, 'UNK':1}

    def _tokenize(self, string):
        if self.lower:
            example = string.strip().lower()
        else:
            example = string.strip()
        return re.findall(self.pattern, example)

    def fit(self, data):
        token_cnts = {}
        for par in data:
            sent_text = nltk.sent_tokenize(par)
            for ex in sent_text:
                example_tokens = self._tokenize(ex)
                for token in example_tokens:
                    if token not in token_cnts:
                        token_cnts[token] = 1
                    else:
                        token_cnts[token] += 1

        index = 2
        for value, key in enumerate(token_cnts):
            if value < self.min_df:
                continue
            self.word_index[key] = index
            if self.wv is not None:
                if key in self.wv:
                    self.embs.append(self.wv[key])
                else:
                    #self.embs.append(np.random.random((300,))*0.01)
                    self.embs.append(np.random.uniform(-1., 1., (300,)))
            else:
                self.embs.append(np.random.uniform(-1., 1., (300,)))
                #self.embs.append(np.random.random((300,))*0.01)
            index += 1

        self.embs = np.array(self.embs)
        del self.wv
        return

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def transform(self, data):
        return_dataset = []
        for par  in data:
            sent_text = nltk.sent_tokenize(par)
            index_sents = []
            for ex in sent_text:
                example = self._tokenize(ex)
                index_example = []
                for token in example:
                    if token in self.word_index:
                        index_example.append(self.word_index[token])
                    else:
                        index_example.append(self.word_index['UNK'])
                index_sents.append(index_example)
            return_dataset.append(index_sents)
        return return_dataset

    def pad_data(self, data):
        max_sents = np.max([len(x) for x in data])
        max_len = np.max([len(x) for y in data for x in y])
        padded_dataset = []
        for par in data:
            pad_sents = []
            for example in par:
                zeros = [0]*(max_len-len(example))
                pad_sents.append(example+zeros)
            for x in range(max_sents-len(par)):
                zeros = [0]*max_len
                pad_sents.append(zeros)
            padded_dataset.append(pad_sents)
        return np.array(padded_dataset)
