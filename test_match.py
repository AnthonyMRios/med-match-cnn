import random
import cPickle
import pickle
from time import time
import sys
import argparse

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from load_data import ProcessData, load_data_file
from metrics import rak, pak
import scipy.sparse as sp

def main():
    parser = argparse.ArgumentParser(description='Test Neural Network.')
    parser.add_argument('--checkpoint_model', help='Checkpoint Model.')
    parser.add_argument('--data_X', help='Test/Validation Data.')
    parser.add_argument('--minibatch_size', type=int, default=32, help='Mini-batch Size.')
    parser.add_argument('--val_minibatch_size', type=int, default=32, help='Mini-batch Size.')
    parser.add_argument('--knn', type=int, default=16, help='KNN Size.')
    parser.add_argument('--train_data_X', help='Training Data.')
    #parser.add_argument('--val_data_X', help='Validation Data.')

    args = parser.parse_args()

    num_neighbors = args.knn
    # Load Checkpoint Model
    with open(args.checkpoint_model,'rb') as out_file:
        chk_pt = pickle.load(out_file)

    # Load & Process Data
    train_txt, train_Y = load_data_file(args.train_data_X)
    #val_txt, val_Y = load_data_file(args.val_data_X)
    test_txt, test_Y = load_data_file(args.data_X)
    X_train = chk_pt['token'].transform(train_txt)
    Y_train = chk_pt['ml_bin'].transform(train_Y)

    X = chk_pt['token'].transform(test_txt)
    Y = chk_pt['ml_bin'].transform(test_Y)
    #Y_val = chk_pt['ml_bin'].transform(val_Y)

    data_processor = chk_pt['token']

    print("Init Model")
    # Init Model
    #from models.cnn import CNN
    #from models.graph_cnn_reg import CNN
    from models.graph_cnn_std_match import CNN
    with open('/home/amri228/naacl_2018/data/mimic2/mimic2_adj_matrix.pkl', 'rb') as in_file:
        adj = cPickle.load(in_file)
    clf = CNN(data_processor.embs, adj, nc=Y.shape[1], de=data_processor.embs.shape[1],
              lr=chk_pt['args'].lr, p_drop=chk_pt['args'].dropout, decay=chk_pt['args'].lr_decay, clip=chk_pt['args'].grad_clip,
              fs=chk_pt['args'].cnn_conv_size, penalty=chk_pt['args'].penalty, train_emb=chk_pt['args'].learn_embeddings)
    clf.__setstate__(chk_pt['model_params'])
    print("CNN: hidden_state: %d word_vec_size: %d lr: %.5f decay: %.6f learn_emb: %s dropout: %.3f num_feat_maps: %d penalty: %.5f conv_widths: %s" % (chk_pt['args'].hidden_state,
                data_processor.embs.shape[1], chk_pt['args'].lr, chk_pt['args'].lr_decay, chk_pt['args'].learn_embeddings, chk_pt['args'].dropout, chk_pt['args'].num_feat_maps, chk_pt['args'].penalty,
                chk_pt['args'].cnn_conv_size))
    sys.stdout.flush()

    train_idxs = list(range(len(X_train)))
    all_features = []
    for start, end in zip(range(0, len(train_idxs), args.val_minibatch_size),
         range(args.val_minibatch_size, len(train_idxs)+args.val_minibatch_size, args.val_minibatch_size)):
        if len(train_idxs[start:end]) == 0:
            continue
        #mini_batch_sample = data_processor.pad_data([X_train[i] for i in train_idxs[start:end]])
        mini_batch_sample = data_processor.pad_data([X_train[i] for i in train_idxs[start:end]], True)
        mini_batch_sample = mini_batch_sample[:,:5000]
        features = clf.mid_feat(mini_batch_sample, np.float32(1.))
        for i in features:
            all_features.append(i)
    all_features = np.array(all_features)

    test_idxs = list(range(len(X)))
    all_test_features = []
    for start, end in zip(range(0, len(test_idxs), args.minibatch_size),
         range(args.minibatch_size, len(test_idxs)+args.minibatch_size+1, args.minibatch_size)):
        if len(test_idxs[start:end]) == 0:
            continue
        mini_batch_sample = data_processor.pad_data([X[i] for i in test_idxs[start:end]], False)
        mini_batch_sample = mini_batch_sample[:,:5000]
        features = clf.mid_feat(mini_batch_sample, np.float32(1.))
        for i in features:
            all_test_features.append(i)
    all_test_features = np.array(all_test_features)

    # Get Predictions
    idxs = list(range(len(X)))
    all_preds = []
    all_pcnt = []
    for start, end in zip(range(0, len(idxs), args.minibatch_size),
            range(args.minibatch_size, len(idxs)+args.minibatch_size, args.minibatch_size)):
        if len(idxs[start:end]) == 0:
            continue
        mini_batch_sample = data_processor.pad_data([X[i] for i in idxs[start:end]])[:,:5000]
        dists = pairwise_distances(all_test_features[idxs[start:end]], all_features)
        rand_idx = []
        for d in dists:
            arg = np.argsort(d)
            tmp = []
            for i in arg: 
                if len(tmp) == num_neighbors:
                    break
                tmp.append(i)
            rand_idx += tmp
            mini_batch_s = data_processor.pad_data([X_train[i] for i in rand_idx], True)
            mini_batch_s = mini_batch_s[:,:5000]


        preds, cost, pcnt = clf.predict_loss(mini_batch_sample, Y[idxs[start:end]].astype('float32'), 
                mini_batch_s, Y_train[rand_idx], np.float32(1.))
        #preds = clf.predict(mini_batch_sample, np.float32(1.))
        for i in preds:
            all_preds.append(i.flatten())
        for i in pcnt.flatten():
            all_pcnt.append(i)

    all_preds = np.array(all_preds)

    new_preds = np.zeros(np.array(all_preds).shape)
    pc = 0
    for row, pcc in zip(np.array(all_preds), all_pcnt):
        for i in np.argsort(row)[::-1][:int(pcc)]:
            new_preds[pc, i] = 1.
        pc += 1
    # Evaluate
    mprcs = precision_recall_fscore_support(Y, (np.array(new_preds)).astype('int32'), average='micro')
    print("Micro: Precision: %.5f Recall: %.5f F1: %.5f" % (mprcs[0], mprcs[1], mprcs[2]))
    sys.stdout.flush()

    maprcs = precision_recall_fscore_support(Y, (np.array(new_preds)).astype('int32'), average='macro')
    print("Macro: Precision: %.5f Recall: %.5f F1: %.5f" % (maprcs[0], maprcs[1], maprcs[2]))
    sys.stdout.flush()

    print("P@8: %.5f" % (pak(Y, np.array(all_preds), 8)))
    sys.stdout.flush()

    print("P@40: %.5f" % (pak(Y, np.array(all_preds), 40)))
    sys.stdout.flush()

    print("R@8: %.5f" % (rak(Y, np.array(all_preds), 8)))
    sys.stdout.flush()

    print("R@40: %.5f" % (rak(Y, np.array(all_preds), 40)))
    sys.stdout.flush()


        


if __name__ == '__main__':
    main()
