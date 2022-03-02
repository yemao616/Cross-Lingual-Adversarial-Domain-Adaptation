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


def prepare_input(df, train_s, val_s, test_s, run, w2v):
    print("\n-------------- Prepare input --------------")
    index_df = blocks_to_index(df, w2v)

    train_df = pd.DataFrame(columns=['sid', 'code', 'label'])
    val_df = pd.DataFrame(columns=['sid', 'code', 'label'])
    test_df = pd.DataFrame(columns=['sid', 'code', 'label'])
    train_len, val_len, test_len = 0, 0, 0

    for row_idx, row in index_df.iterrows():
        cur_id = row['sid']
        tmpx = row['blocks_seq']
        tmpy = int(row['label'])

        if cur_id in train_s:
            train_df = train_df.append({'sid': cur_id, 'code': tmpx, 'label': tmpy}, ignore_index=True)
            train_len += 1

        elif cur_id in val_s:
            val_df = val_df.append({'sid': cur_id, 'code': tmpx, 'label': tmpy}, ignore_index=True)
            val_len += 1

        elif cur_id in test_s:
            test_df = test_df.append({'sid': cur_id, 'code': tmpx, 'label': tmpy}, ignore_index=True)
            test_len += 1
        else:
            pass

    print("Data is ready for the [{}] resample run".format(run + 1))

    args.train_data = train_df
    args.val_data = val_df
    args.test_data = test_df
    return train_len, val_len, test_len


if __name__ == '__main__':
    args = config_parser().parse_args('--name ASTNN --lang Snap --batch 20 --epochs 50'.split())
    # args = config_parser().parse_args()

    all_data = pd.read_pickle('./data/{}/parsed_prob.pkl'.format(args.language))
    semester = all_data['semester'].unique().tolist()
    semester.sort(key=lambda x: (-int(x[-1]), x[0]), reverse=True)
    res = pd.DataFrame(index=semester+['overall'],
                       columns=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1_score', 'Confusion_matrix'])

    all_true, all_pred = [], []
    for test_idx in range(1, len(semester)):
        test_semester = semester[test_idx]
        train_semester = semester[:test_idx]
        print("\ntest semester:", test_semester, "train semester:", train_semester)

        data = all_data.loc[all_data['semester'].isin(train_semester + [test_semester]), ]
        train_val_sid = np.array(data.loc[data['semester'].isin(train_semester), 'sid'].values.tolist())
        test_sid = data.loc[data['semester'].isin([test_semester]), 'sid'].values.tolist()

        # ---- load SnapJava w2v
        word2vec = Word2Vec.load('./embeddings/w2v_SnapJava_{}'.format(args.embedding_dim)).wv
        # # ---- load all semester w2v
        # word2vec = Word2Vec.load('./embeddings/w2v_{}_{}'.format(args.language, args.embedding_dim)).wv
        # # ---- load semester-specific w2v
        # word2vec = Word2Vec.load(
        #     './embeddings/w2v_{}_{}_{}'.format(args.language, args.embedding_dim, test_semester)).wv
        max_tokens = word2vec.vectors.shape[0]
        embeddings = np.zeros((max_tokens + 1, args.embedding_dim), dtype="float32")
        embeddings[:max_tokens] = word2vec.vectors
        args.pretrained_embedding, args.vocab_size = embeddings, max_tokens + 1

        """
        for each semester fold, run 5-cv
        """
        best_acc = 0.0
        pred, actual = [], []
        kf = KFold(n_splits=5, shuffle=True)
        for run_id, (train_idx, val_idx) in enumerate(kf.split(train_val_sid)):
            train_sid = train_val_sid[train_idx].tolist()
            val_sid = train_val_sid[val_idx].tolist()

            train_num, val_num, test_num = prepare_input(df=data, run=run_id,
                                                         train_s=train_sid, val_s=val_sid, test_s=test_sid,
                                                         w2v=word2vec)
            print("training size:", train_num, "validation size:", val_num, "testing size:", test_num)

            trainer = Trainer(args)
            test_pred, test_actual = trainer.run(info=test_semester, test_inputs=trainer.val_data)
            cur_acc = accuracy_score(test_actual, test_pred)

            # record best run among 10-cv
            if cur_acc > best_acc:
                best_acc = cur_acc
                pred, actual = test_pred, test_actual

        # save best result for each semester
        res = save_result(res, test_semester, pred, actual)
        res.to_csv('./result/res_{}_{}.csv'.format(args.name, args.language))

        # append predictions and labels
        all_true.extend(actual)
        all_pred.extend(pred)

    # save overall result
    res = save_result(res, 'overall', all_pred, all_true)
    res.to_csv('./result/res_{}_{}.csv'.format(args.name, args.language))
