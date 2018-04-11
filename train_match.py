import sys
import os
import random
import pickle
import cPickle
import argparse
from time import time

import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import f1_score

from load_data import ProcessData, load_data_file, ProcessHierData
from label_bin import CustomLabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics.pairwise import pairwise_distances
from metrics import rak, pak

def main():
    parser = argparse.ArgumentParser(description='Train Neural Network.')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of updates to make.')
    parser.add_argument('--num_models', type=int, default=5, help='Number of updates to make.')
    parser.add_argument('--word_vectors', default=None, help='Word vecotors filepath.')
    parser.add_argument('--labels', default='/home/amri228/naacl_2018/data/mimic2/all_labels.txt',
                        help='All Labels.')
    parser.add_argument('--checkpoint_dir', default='./experiments/exp1/checkpoints/',
                        help='Checkpoint directory.')
    parser.add_argument('--checkpoint_name', default='checkpoint',
                        help='Checkpoint File Name.')
    parser.add_argument('--hidden_state', type=int, default=2048, help='hidden layer size.')
    parser.add_argument('--learn_embeddings', type=bool, default=True, help='Learn Embedding Parameters.')
    parser.add_argument('--min_df', type=int, default=5, help='Min word count.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate.')
    parser.add_argument('--penalty', type=float, default=0.0, help='Regularization Parameter.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout Value.')
    parser.add_argument('--lr_decay', type=float, default=1e-6, help='Learning Rate Decay.')
    parser.add_argument('--minibatch_size', type=int, default=2, help='Mini-batch Size.')
    parser.add_argument('--val_minibatch_size', type=int, default=2, help='Val Mini-batch Size.')
    parser.add_argument('--model_type', help='Neural Net Architecutre.')
    parser.add_argument('--train_data_X', help='Training Data.')
    parser.add_argument('--val_data_X', help='Validation Data.')
    parser.add_argument('--seed', default=9999, type=int, help='Random Seed.')
    parser.add_argument('--grad_clip', type=float, default=None, help='Gradient Clip Value.')
    parser.add_argument('--cnn_conv_size', nargs='+', type=int, default=[3], help='CNN Covolution Sizes (widths)')
    parser.add_argument('--num_feat_maps', default=300, type=int, help='Number of CNN Feature Maps.')
    parser.add_argument('--num_att', default=30, type=int, help='Number of Heads.')
    parser.add_argument('--num_support', default=16, type=int, help='Number nearest neighbors to sample for each input instance.')

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load & Process Data
    train_txt, train_Y = load_data_file(args.train_data_X)
    val_txt, val_Y = load_data_file(args.val_data_X)

    data_processor = ProcessData(args.word_vectors, lower=True, min_df=args.min_df)
    X_train = data_processor.fit_transform(train_txt)
    print 'AVG LEN:', np.mean([len(x) for x in X_train])
    sys.stdout.flush()
    X_val = data_processor.transform(val_txt)

    labels = []
    with open(args.labels,'r') as in_file:
        for row in in_file:
            labels.append(row.strip())
    lookup = set(labels)

    ml_vec = MultiLabelBinarizer(classes=labels)
    ml_vec.fit(train_Y)
    Y_train = ml_vec.transform(train_Y)
    print Y_train.shape, 'SHAPE'
    print 'max:', Y_train.sum(axis=0).max(), Y_train.sum(axis=0).min()
    Y_val = ml_vec.transform(val_Y)

    print("Init Model")
    sys.stdout.flush()
    # Init Model
    if args.model_type == 'cnn':
        from models.match_cnn_reg import CNN
        clf = CNN(data_processor.embs, nc=Y_train.shape[1], de=data_processor.embs.shape[1],
                  lr=args.lr, p_drop=args.dropout, decay=args.lr_decay, clip=args.grad_clip,
                  fs=args.cnn_conv_size, penalty=args.penalty, train_emb=args.learn_embeddings, num_heads=args.num_att)
        print("CNN: hidden_state: %d word_vec_size: %d lr: %.5f decay: %.6f learn_emb: %s dropout: %.3f num_feat_maps: %d penalty: %.5f conv_widths: %s" % (args.hidden_state,
                    data_processor.embs.shape[1], args.lr, args.lr_decay, args.learn_embeddings, args.dropout, args.num_feat_maps, args.penalty,
                    args.cnn_conv_size))
    else:
        raise ValueError('Incorrect Model Specified')

    print("Training Model")
    sys.stdout.flush()
    train_idxs = list(range(len(X_train)))
    val_idxs = list(range(len(X_val)))
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

    # Train Model
    best_val_f1 = 0
    best_macro_val_f1 = 0
    for epoch in range(1, args.num_epochs+1):
        mean_loss = []
        mean_micro_f1 = []
        mean_macro_f1 = []
        random.shuffle(train_idxs)
        epoch_t0 = time()
        print "MINIBATCH SIZE: %d" % (args.minibatch_size)
        print len(train_idxs)
        for start, end in zip(range(0, len(train_idxs), args.minibatch_size),
             range(args.minibatch_size, len(train_idxs)+args.minibatch_size, args.minibatch_size)):
            if len(train_idxs[start:end]) == 0:
                continue
            #mini_batch_sample = data_processor.pad_data([X_train[i] for i in train_idxs[start:end]])
            mini_batch_sample = data_processor.pad_data([X_train[i] for i in train_idxs[start:end]], True)
            mini_batch_sample = mini_batch_sample[:,:5000] 

            dists = pairwise_distances(all_features[train_idxs[start:end]], all_features)
            #dists = np.dot(all_features[val_idxs[start:end]], all_features.T)
            rand_idx = []
            check = set(train_idxs[start:end])
            sY = []
            for d in dists:
                arg = np.argsort(d)
                tmp = []
                for i in arg: 
                    if len(tmp) == args.num_support:
                        break
                    if i not in check:
                        tmp.append(i)
                #tmp2 = random.sample(tmp, 16)
                rand_idx += tmp
            mini_batch_s = data_processor.pad_data([X_train[i] for i in rand_idx], True)
            mini_batch_s = mini_batch_s[:,:5000] 

            cost, preds, new_h, new_h2, pcnt = clf.train_batch(mini_batch_sample,
                    Y_train[train_idxs[start:end]].astype('float32'),
                    mini_batch_s,
                    Y_train[rand_idx].astype('float32'),
                    np.float32(0.))
            costc = clf.train_count(mini_batch_sample,
                    Y_train[train_idxs[start:end]].astype('float32').sum(axis=1),
                    np.float32(0.))

            for i, h in zip(train_idxs[start:end], new_h):
                all_features[i] = h
            for i, h in zip(rand_idx, new_h2):
                all_features[i] = h

            new_preds = np.zeros(np.array(preds).shape)
            pc = 0
            for row, pcc in zip(np.array(preds), pcnt):
                for i in np.argsort(row)[::-1][:int(pcc)]:
                    new_preds[pc, i] = 1.
                pc += 1
            #micro_f1 = f1_score((Y_train[train_idxs[start:end]]>0.5).astype('int32'), (np.array(new_preds, dtype='float32')>0.5).astype('int32'), average='micro')
            #macro_f1 = f1_score((Y_train[train_idxs[start:end]]>0.5).astype('int32'), (np.array(new_preds, dtype='float32')>0.5).astype('int32'), average='macro')
            micro_f1 = f1_score((Y_train[train_idxs[start:end]]>0.5).astype('int32'), (np.array(new_preds, dtype='float32')).astype('int32'), average='micro')
            macro_f1 = f1_score((Y_train[train_idxs[start:end]]>0.5).astype('int32'), (np.array(new_preds, dtype='float32')).astype('int32'), average='macro')
            pa8 = pak(Y_train[train_idxs[start:end]], np.array(preds).astype('float32'), 8)
            pa40 = pak(Y_train[train_idxs[start:end]], np.array(preds).astype('float32'), 40)
            ra8 = rak(Y_train[train_idxs[start:end]], np.array(preds).astype('float32'), 8)
            ra40 = rak(Y_train[train_idxs[start:end]], np.array(preds).astype('float32'), 40)
            mean_micro_f1.append(micro_f1)
            mean_macro_f1.append(macro_f1)
            mean_loss.append(cost)
            sys.stdout.write("Epoch: %d train_avg_loss: %.4f train_avg_micro_f1: %.4f train_avg_macro_f1: %.4f sum1: %d sum2: %d p@8: %.4f p@40: %.4f r@8: %.4f r@40: %.4f\n" %
                    (epoch, np.mean(mean_loss), np.mean(mean_micro_f1), np.mean(mean_macro_f1), Y_train[train_idxs[start:end]].sum(), np.array(preds).sum(), pa8, pa40, ra8, ra40))
            sys.stdout.flush()

        # Validate Model
        all_val_features = []
        for start, end in zip(range(0, len(val_idxs), args.minibatch_size),
             range(args.minibatch_size, len(val_idxs)+args.minibatch_size+1, args.minibatch_size)):
            if len(val_idxs[start:end]) == 0:
                continue
            mini_batch_sample = data_processor.pad_data([X_val[i] for i in val_idxs[start:end]], False)
            mini_batch_sample = mini_batch_sample[:,:5000] 
            features = clf.mid_feat(mini_batch_sample, np.float32(1.))
            for i in features:
                all_val_features.append(i)
        all_val_features = np.array(all_val_features)

        final_preds = []
        val_loss = []
        all_pcnt = []
        for start, end in zip(range(0, len(val_idxs), args.val_minibatch_size),
             range(args.val_minibatch_size, len(val_idxs)+args.val_minibatch_size, args.val_minibatch_size)):
            if len(val_idxs[start:end]) == 0:
                continue
            mini_batch_sample = data_processor.pad_data([X_val[i] for i in val_idxs[start:end]], False)
            mini_batch_sample = mini_batch_sample[:,:5000] 

            dists = pairwise_distances(all_val_features[val_idxs[start:end]], all_features)
            #dists = np.dot(all_val_features[val_idxs[start:end]], all_features.T)
            rand_idx = []
            for d in dists:
                arg = np.argsort(d)
                tmp = []
                for i in arg: 
                    if len(tmp) == args.num_support:
                        break
                    tmp.append(i)
                rand_idx += tmp
            mini_batch_s = data_processor.pad_data([X_train[i] for i in rand_idx], True)
            mini_batch_s = mini_batch_s[:,:5000] 

            preds, cost, pcnt = clf.predict_loss(mini_batch_sample, Y_val[val_idxs[start:end]].astype('float32'), 
                    mini_batch_s, Y_train[rand_idx], np.float32(1.))
            for x in preds:
                final_preds.append(x.flatten())
            for i in pcnt.flatten():
                all_pcnt.append(i)
            val_loss.append(cost)

        new_preds = np.zeros(np.array(final_preds).shape)
        pc = 0
        for row, pcc in zip(np.array(final_preds), all_pcnt):
            for i in np.argsort(row)[::-1][:int(pcc)]:
                new_preds[pc, i] = 1.
            pc += 1
        #micro_f1 = f1_score(Y_val.astype('int32'), (np.array(final_preds).astype('float32')>0.5).astype('int32'), average='micro')
        #macro_f1 = f1_score(Y_val.astype('int32'), (np.array(final_preds).astype('float32')>0.5).astype('int32'), average='macro')
        micro_f1 = f1_score(Y_val.astype('int32'), (np.array(new_preds).astype('float32')).astype('int32'), average='micro')
        macro_f1 = f1_score(Y_val.astype('int32'), (np.array(new_preds).astype('float32')).astype('int32'), average='macro')
        pa8 = pak(Y_val, np.array(final_preds).astype('float32'), 8)
        pa40 = pak(Y_val, np.array(final_preds).astype('float32'), 40)
        ra8 = rak(Y_val, np.array(final_preds).astype('float32'), 8)
        ra40 = rak(Y_val, np.array(final_preds).astype('float32'), 40)

        sys.stdout.write("epoch: %d val_loss %.4f val_micro_f1: %.4f val_macro_f1: %.4f train_avg_loss: %.4f train_avg_f1: %.4f time: %.1f p@8: %.4f p@40: %.4f r@8: %.4f r@40: %.4f\n" %
                (epoch, np.mean(val_loss), micro_f1, macro_f1, np.mean(mean_loss), np.mean(mean_micro_f1), time()-epoch_t0, pa8, pa40, ra8, ra40))
        sys.stdout.flush()

        # Checkpoint Model
        if micro_f1 > best_val_f1:
            best_val_f1 = micro_f1
            with open(os.path.abspath(args.checkpoint_dir)+'/'+args.checkpoint_name+'_micro_graph2.pkl','wb') as out_file:
                pickle.dump({'model_params':clf.__getstate__(), 'token':data_processor,
                             'ml_bin':ml_vec, 'args':args, 'last_train_avg_loss': np.mean(mean_loss),
                             'last_train_avg_f1':np.mean(mean_micro_f1), 'val_f1':micro_f1}, out_file, pickle.HIGHEST_PROTOCOL)

        if macro_f1 > best_macro_val_f1:
            best_macro_val_f1 = macro_f1
            with open(os.path.abspath(args.checkpoint_dir)+'/'+args.checkpoint_name+'_macro_graph2.pkl','wb') as out_file:
                pickle.dump({'model_params':clf.__getstate__(), 'token':data_processor,
                             'ml_bin':ml_vec, 'args':args, 'last_train_avg_loss': np.mean(mean_loss),
                             'last_train_avg_f1':np.mean(mean_micro_f1), 'val_f1':micro_f1}, out_file, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
