from theano import tensor as T
import theano
import numpy as np
from theano.tensor.nnet import conv2d
from theano.tensor.signal.pool import pool_2d as max_pool_2d
from theano.tensor.shared_randomstreams import RandomStreams
srng2 = RandomStreams(seed=234)

from utils import *

class CNN(object):
    ''' CNN Model (http://www.aclweb.org/anthology/D14-1181)
    '''
    def __init__(self, emb, nf=300, nc=2, de=300, p_drop=0.5, fs=[3,4,5], penalty=0,
            lr=0.001, decay=0., clip=None, train_emb=True, num_heads=32):
        ''' Init Experimental CNN model.

            Args:
            emb: Word embeddings matrix (num_words x word_dimension)
            nc: Number of classes
            de: Dimensionality of word embeddings
            p_drop: Dropout probability
            fs: Convolution filter width sizes
            penalty: l2 regularization param
            lr: Initial learning rate
            decay: Learning rate decay parameter
            clip: Gradient Clipping parameter (None == don't clip)
            train_emb: Boolean if the embeddings should be trained
        '''
        self.emb = theano.shared(name='Words',
            value=as_floatX(emb))

        self.filter_w = []
        self.filter_b = []
        for filter_size in fs:
            self.filter_w.append(theano.shared(
                value=he_normal((nf, 1, filter_size, de))
                .astype('float32')))
            self.filter_b.append(theano.shared(
                value=np.zeros((nf,)).astype('float32')))

        self.w_h = theano.shared(value=he_normal((nf*len(fs), nf*len(fs), num_heads)).astype('float32'))
        self.b_h = theano.shared(value=he_normal((nf*len(fs), num_heads)).astype('float32'))

        self.w_h3 = theano.shared(value=he_normal((nf*len(fs), nf*len(fs))).astype('float32'))
        self.b_h3 = theano.shared(value=he_normal((nf*len(fs), )).astype('float32'))

        self.w_count = theano.shared(value=he_normal((nf*len(fs), 1)).astype('float32'))
        self.b_count = theano.shared(value=as_floatX(np.zeros((1,))))

        self.b_o = theano.shared(value=as_floatX(np.zeros((nc,))))
        self.w_o = theano.shared(value=he_normal((nf*len(fs)*num_heads, nc)).astype('float32'))
        #self.w_ho2 = theano.shared(value=he_normal((nf*len(fs), 128, 10)).astype('float32'))

        #self.b_o2 = theano.shared(value=as_floatX(np.zeros((nc,))))
        self.w_o2 = theano.shared(value=he_normal((nf*len(fs), nc)).astype('float32'))
        #self.b_o2 = theano.shared(value=as_floatX(np.zeros((nc,))))

        #self.params = [self.emb, self.w_h, self.b_h, self.w_o, self.b_o, self.w_h3, self.b_h3, self.w_o2, self.b_o2, self.w_att]
        #self.params = [self.emb, self.w_h, self.b_h, self.w_o, self.b_o, self.w_h3, self.b_h3, self.w_att, self.w_ho, self.w_ho2]
        #self.params = [self.emb, self.w_h, self.b_h, self.w_o, self.b_o, self.w_h3, self.b_h3, self.w_att, self.w_ho, self.w_ho2]
        #self.params = [self.emb, self.w_h, self.b_h, self.w_o, self.b_o, self.w_h3, self.b_h3, self.w_att, self.w_ho2]
        self.params = [self.emb, self.w_o, self.b_o, self.w_h, self.b_h, self.w_h3, self.b_h3, self.w_o2]
        self.params_counts = [self.w_count, self.b_count]

        for w, b in zip(self.filter_w, self.filter_b):
            self.params.append(w)
            self.params.append(b)

        dropout_switch = T.fscalar('dropout_switch')

        idxs = T.matrix()
        x_word = self.emb[T.cast(idxs.flatten(), 'int32')].reshape((idxs.shape[0], 1, idxs.shape[1], de))
        mask = T.neq(idxs, 0)*as_floatX(1.)
        x_word = x_word*mask.dimshuffle(0, 'x', 1, 'x')
        Y = T.matrix('Y')
        Y_counts = T.vector('Yc')

        sidxs = T.matrix()
        sx_word = self.emb[T.cast(sidxs.flatten(), 'int32')].reshape((sidxs.shape[0], 1, sidxs.shape[1], de))
        smask = T.neq(sidxs, 0)*as_floatX(1.)
        sx_word = sx_word*smask.dimshuffle(0, 'x', 1, 'x')
        Ys = T.matrix('Ys')

        l1_w_all = []
        for w, b, width in zip(self.filter_w, self.filter_b, fs):
            l1_w = conv2d(x_word, w, input_shape=(None,1,None,de), filter_shape=(nf, 1, width, de))
            l1_w = T.nnet.relu(l1_w+ b.dimshuffle('x', 0, 'x', 'x'))
            l1_w = T.max(l1_w, axis=2).flatten(2)
            l1_w_all.append(l1_w)

        sl1_w_all = []
        for w, b, width in zip(self.filter_w, self.filter_b, fs):
            sl1_w = conv2d(sx_word, w, input_shape=(None,1,None,de), filter_shape=(nf, 1, width, de))
            sl1_w = T.nnet.relu(sl1_w + b.dimshuffle('x', 0, 'x', 'x'))
            sl1_w = T.max(sl1_w, axis=2).flatten(2)
            sl1_w_all.append(sl1_w)

        h = T.concatenate(l1_w_all, axis=1)
        h2 = T.nnet.relu(h.dot(self.w_h) + self.b_h).dimshuffle(0,2,1)
        #h = dropout(h, dropout_switch, p_drop)
        sh1 = T.concatenate(sl1_w_all, axis=1)
        #Ys2 = Ys * 1./(Ys.sum(axis=0)+1e-6)
        #sh = T.dot(sh.T, Ys2).T
        sh = T.nnet.relu(T.dot(sh1, self.w_h3) + self.b_h3)
        #sh = dropout(sh, dropout_switch, p_drop)

        def attention(h1, s1, Ys1):
            squared_euclidean_distances = (h1 ** 2).sum(1).reshape((h1.shape[0], 1)) + (s1 ** 2).sum(1).reshape((1, s1.shape[0])) - 2 * h1.dot(s1.T)
            scores = T.nnet.softmax(squared_euclidean_distances)
            pyx = T.dot(scores, Ys1).mean(axis=0)
            feats = T.dot(scores, s1)
            #pyx = T.clip(pyx, 1e-7, 1.-1e-7)
            return pyx, feats.flatten()

        [l_pyx, feats], updates = theano.scan(attention, sequences=[h2],
                non_sequences=[sh, Ys], outputs_info=[None, None])

        #h3 = dropout(h, dropout_switch, p_drop)
        #feats2 = dropout(feats, dropout_switch, p_drop)
        #h3 = T.tanh(h.dot(self.w_ho2)).dimshuffle(0,2,1)
        #self.w_ho = theano.shared(value=he_normal((nf*len(fs)*num_heads, 128, 5)).astype('float32'))
        pyx = T.nnet.sigmoid(T.dot(feats, self.w_o) + T.dot(h, self.w_o2) + self.b_o)
        pyx = T.clip(pyx, 1e-7, 1.-1e-7)
        #weights = (Y * 9.) + 1.
        #L = (T.nnet.nnet.binary_crossentropy(pyx, Y)*weights).mean()
        L = (T.nnet.nnet.binary_crossentropy(pyx, Y)).mean()

        count = T.nnet.relu(T.dot(h, self.w_count) + self.b_count).flatten()
        L_count = ((count - Y_counts)**2).sum()
        updates = Adam(L, self.params, lr2=lr, clip=clip)
        updates_count = Adam(L_count, self.params_counts, lr2=lr, clip=clip)

        self.mid_feat = theano.function([idxs, dropout_switch], h, allow_input_downcast=True, on_unused_input='ignore')
        self.train_batch = theano.function([idxs, Y, sidxs, Ys, dropout_switch], [L, pyx, h, sh1, count], updates=updates, allow_input_downcast=True, on_unused_input='ignore')
        self.train_count = theano.function([idxs, Y_counts, dropout_switch], L_count, updates=updates_count, allow_input_downcast=True, on_unused_input='ignore')

        self.predict = theano.function([idxs, sidxs, Ys, dropout_switch],
                outputs=pyx, allow_input_downcast=True, on_unused_input='ignore')
        self.predict_proba = theano.function([idxs, sidxs, Ys, dropout_switch], outputs=pyx, allow_input_downcast=True, on_unused_input='ignore')
        self.predict_loss = theano.function([idxs, Y, sidxs, Ys, dropout_switch], [pyx, L, count], allow_input_downcast=True, on_unused_input='ignore')

    def __getstate__(self):
        return [x.get_value() for x in self.params] + [x.get_value() for x in self.params_counts]

    def __setstate__(self, data):
        for w, p in zip(data, self.params+self.params_counts):
            p.set_value(w)
        return

    def __setstate2__(self, data):
        self.emb.set_value(data[0])
        #self.w_o.set_value(data[1])
        self.b_o.set_value(data[2])
        cnt = 3
        for f1 in self.filter_w:
            f1.set_value(data[cnt])
            cnt += 1
        for f1 in self.filter_b:
            f1.set_value(data[cnt])
            cnt += 1

