# EMR Coding with Semi-Parametric Multi-Head Matching Networks

This repo contains code for our 2018 NAACL paper titled "EMR Coding with Semi-Parametric Multi-Head Matching Networks".

**Note:** Examples of the data format can be found in the data/ folder.

## Usage
### Training

```
python train_match.py --num_epochs 25 --word_vectors 'gensim_w2v_pubmed' --model_type cnn --train_data_X './data/train_data.json' --val_data_X './data/dev_data.json' --checkpoint_dir './checkpoints' --num_feat_maps 300 --grad_clip 3 --min_df 5 --lr 0.0001 --penalty 0.0000 --dropout 0.5 --lr_decay 0.0000 --cnn_conv_size 3 4 5  --checkpoint_name my_model_name
```

```
usage: train_match.py [-h] [--num_epochs NUM_EPOCHS] [--num_models NUM_MODELS]
                      [--word_vectors WORD_VECTORS] [--labels LABELS]
                      [--checkpoint_dir CHECKPOINT_DIR]
                      [--checkpoint_name CHECKPOINT_NAME]
                      [--hidden_state HIDDEN_STATE]
                      [--learn_embeddings LEARN_EMBEDDINGS] [--min_df MIN_DF]
                      [--lr LR] [--penalty PENALTY] [--dropout DROPOUT]
                      [--lr_decay LR_DECAY] [--minibatch_size MINIBATCH_SIZE]
                      [--val_minibatch_size VAL_MINIBATCH_SIZE]
                      [--model_type MODEL_TYPE] [--train_data_X TRAIN_DATA_X]
                      [--val_data_X VAL_DATA_X] [--seed SEED]
                      [--grad_clip GRAD_CLIP]
                      [--cnn_conv_size CNN_CONV_SIZE [CNN_CONV_SIZE ...]]
                      [--num_feat_maps NUM_FEAT_MAPS] [--num_att NUM_ATT]
                      [--num_support NUM_SUPPORT]

Train Neural Network.

optional arguments:
  -h, --help            show this help message and exit
  --num_epochs NUM_EPOCHS
                        Number of updates to make.
  --num_models NUM_MODELS
                        Number of updates to make.
  --word_vectors WORD_VECTORS
                        Word vecotors filepath.
  --labels LABELS       All Labels.
  --checkpoint_dir CHECKPOINT_DIR
                        Checkpoint directory.
  --checkpoint_name CHECKPOINT_NAME
                        Checkpoint File Name.
  --hidden_state HIDDEN_STATE
                        hidden layer size.
  --learn_embeddings LEARN_EMBEDDINGS
                        Learn Embedding Parameters.
  --min_df MIN_DF       Min word count.
  --lr LR               Learning Rate.
  --penalty PENALTY     Regularization Parameter.
  --dropout DROPOUT     Dropout Value.
  --lr_decay LR_DECAY   Learning Rate Decay.
  --minibatch_size MINIBATCH_SIZE
                        Mini-batch Size.
  --val_minibatch_size VAL_MINIBATCH_SIZE
                        Val Mini-batch Size.
  --model_type MODEL_TYPE
                        Neural Net Architecutre.
  --train_data_X TRAIN_DATA_X
                        Training Data.
  --val_data_X VAL_DATA_X
                        Validation Data.
  --seed SEED           Random Seed.
  --grad_clip GRAD_CLIP
                        Gradient Clip Value.
  --cnn_conv_size CNN_CONV_SIZE [CNN_CONV_SIZE ...]
                        CNN Covolution Sizes (widths)
  --num_feat_maps NUM_FEAT_MAPS
                        Number of CNN Feature Maps.
  --num_att NUM_ATT     Number of Heads.
  --num_support NUM_SUPPORT
                        Number nearest neighbors to sample for each input
                        instance.
```

### Testing

The file "test_match.py" provides an example on how to run and evaluate our method.

```
python test_match.py --data_X './data/test_data.json' --checkpoint_model './checkpoints/my_model_name.pkl' --train_data_X './data/train_data.json' --minibatch_size 3 --knn 8 --val_minibatch_size 3
```

```
usage: test_match.py [-h] [--checkpoint_model CHECKPOINT_MODEL]
                     [--data_X DATA_X] [--minibatch_size MINIBATCH_SIZE]
                     [--val_minibatch_size VAL_MINIBATCH_SIZE] [--knn KNN]
                     [--train_data_X TRAIN_DATA_X]

Test Neural Network.

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint_model CHECKPOINT_MODEL
                        Checkpoint Model.
  --data_X DATA_X       Test/Validation Data.
  --minibatch_size MINIBATCH_SIZE
                        Mini-batch Size.
  --val_minibatch_size VAL_MINIBATCH_SIZE
                        Mini-batch Size.
  --knn KNN             KNN Size.
  --train_data_X TRAIN_DATA_X
                        Training Data.
```

## Acknowledgements

> Anthony Rios and Ramakanth Kavuluru. "EMR Coding with Semi-Parametric Multi-Head Matching Networks". NAACL 2018

```
@inproceedings{arios2018emrmatch,
  title={EMR Coding with Semi-Parametric Multi-Head Matching Networks},
  author={Rios, Anthony and Kavuluru, Ramakanth},
  booktitle={Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  year={2018}
}
```

Written by Anthony Rios (anthonymrios at gmail dot com)
