import torch
import argparse
import random
import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score, precision_score
from trainer import Trainer
from utils.util import config_parser, blocks_to_index, save_result


def prepare_input(df, train_s, test_s, run, w2v):
    print("\n-------------- Prepare input --------------")
    index_df = blocks_to_index(df, w2v)

    train_df = pd.DataFrame(columns=['sid', 'code', 'label'])
    test_df = pd.DataFrame(columns=['sid', 'code', 'label'])
    train_len, test_len = 0, 0

    for row_idx, row in index_df.iterrows():
        cur_id = row['sid']
        tmpx = row['blocks_seq']
        tmpy = int(row['label'])

        if cur_id in train_s:
            train_df = train_df.append({'sid': cur_id, 'code': tmpx, 'label': tmpy}, ignore_index=True)
            train_len += 1

        elif cur_id in test_s:
            test_df = test_df.append({'sid': cur_id, 'code': tmpx, 'label': tmpy}, ignore_index=True)
            test_len += 1
        else:
            pass

    print("Data is ready for the [{}] resample run".format(run))

    args.train_data = train_df
    args.test_data = test_df
    return train_len, test_len


if __name__ == '__main__':
    cmd = '--name ASTNN --lang Java --batch 20 --epochs 50 --embedding_dim 64 --encode_dim 64'
    args = config_parser().parse_args(cmd.split())
    # args = config_parser().parse_args()

    # ---- load data
    all_data = pd.read_pickle('./data/{}/parsed_prob.pkl'.format(args.language))

    # ---- load w2v
    word2vec = Word2Vec.load('./embeddings/temporal/w2v_SnapJava_{}'.format(args.embedding_dim)).wv
    max_tokens = word2vec.vectors.shape[0]
    embeddings = np.zeros((max_tokens + 1, args.embedding_dim), dtype="float32")
    embeddings[:max_tokens] = word2vec.vectors
    args.pretrained_embedding, args.vocab_size = embeddings, max_tokens + 1

    n_fold = 5
    res = pd.DataFrame(index=list(range(n_fold))+['overall'],
                       columns=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1_score', 'Confusion_matrix'])

    all_true, all_pred = [], []
    kf = KFold(n_splits=n_fold, shuffle=True)
    for cur_fold, (train_sid, test_id) in enumerate(kf.split(all_data)):
        cur_fold += 1
        print("\ncurrent fold:", cur_fold)

        train_sid = all_data.loc[train_sid, 'sid'].values.tolist()
        test_sid = all_data.loc[test_id, 'sid'].values.tolist()

        train_num, test_num = prepare_input(df=all_data, run=cur_fold, train_s=train_sid, test_s=test_sid, w2v=word2vec)
        print("training size:", train_num, "testing size:", test_num)

        trainer = Trainer(args)
        pred, actual = trainer.run(info='fold'+str(cur_fold))

        # save best result for each semester
        res = save_result(res, cur_fold, pred, actual)
        res.to_csv('./result/res_{}_{}.csv'.format(args.name, args.language))

        # append predictions and labels
        all_true.extend(actual)
        all_pred.extend(pred)

    # save overall result
    res = save_result(res, 'overall', all_pred, all_true)
    res.to_csv('./result/res_{}_{}.csv'.format(args.name, args.language))
