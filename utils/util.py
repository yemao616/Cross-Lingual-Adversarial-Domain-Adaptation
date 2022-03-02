from visdom import Visdom
import numpy as np
import pandas as pd
import argparse
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score, precision_score


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='model name')
    parser.add_argument('--language', type=str, default='', help='programming language for input data')
    parser.add_argument('--data', type=str, default='', metavar='D', help='input data')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--sequential', action='store_true', default=False, help='enables sequential ASTNN')
    parser.add_argument('-gpu', '--use_gpu', action='store_true', default=False, help='enables gpu')
    parser.add_argument('--cuda', type=int, default=3, metavar='D', help='cuda device (default: 3)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('-ed', '--embedding_dim', type=int, default=128, metavar='N',
                        help='word embedding dim (default: 128)')
    parser.add_argument('-encode', '--encode_dim', type=int, default=128, metavar='N',
                        help='encode dim for AST (default: 128)')
    parser.add_argument('-hidden', '--hidden_dim', type=int, default=100, metavar='N',
                        help='hidden dim for RNN (default: 100)')
    parser.add_argument('--pretrained_embedding', default=None)
    parser.add_argument('--vocab_size', type=int, default=1)
    parser.add_argument('-label', '--label_size', type=int, default=1, metavar='N', help='label size (default: 1)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('-batch', '--batch_size', dest='batch_size', type=int, default=32, metavar='N',
                        help='batch size for training/testing (default: 32)')
    parser.add_argument('--log_interval', type=int, default=2, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument("-v", "--verbose", dest="verbose_mode", type=int, default=1,
                        help="verbose mode (should be in {0,1}).")
    parser.add_argument("--logverbose", type=int, default=0,
                        help="verbose mode for train/test log (should be in {0,1}).")
    parser.add_argument("-log", "--log_path", dest="log_path", metavar="FILE", default=None,
                        help="path to store logs into. if not given logs are not saved to file.")

    parser.add_argument('--train_data', default='', type=str, help='training data (default: None)')
    parser.add_argument('--val_data', default='', type=str, help='validation data (default: None)')
    parser.add_argument('--test_data', default='', type=str, help='testing data (default: None)')
    parser.add_argument('--expert', action='store_true', default=False, help='enables expert features or not')
    parser.add_argument('--time', action='store_true', default=False, help='enables time features or not')


    # ------------------------------ for Sequential-ASTNN only ------------------------------ #
    parser.add_argument('--load_pretrain', action='store_true', default=False, help='load pretrained-ASTNN')
    parser.add_argument('--freeze_astnn', action='store_true', default=False, help='freeze whole ASTNN')
    parser.add_argument('--freeze_encoder', action='store_true', default=False, help='freeze ASTNN Encoder')
    parser.add_argument('--freeze_gru', action='store_true', default=False, help='freeze ASTNN GRU')
    parser.add_argument('--run', type=int, help='load pretrain weight from early mins run,'
                                                'only works when --load_pretrain is true')

    # ------------------------------ for CLASTNN only ------------------------------ #
    parser.add_argument('--with_target_domain', action='store_true', default=False,
                        help='enables input for target domain or not (default: False)')

    parser.add_argument('--domain_size', type=int, default=1, metavar='N', help='domain size (default: 1)')
    parser.add_argument('--m', type=str, default='Snap', help='source domain 1 (default: Snap)')
    parser.add_argument('--n', type=str, default='Java', help='source domain 2 (default: Java)')
    parser.add_argument('--m_train_data', type=str, default='', help='training data for source domain 1')
    parser.add_argument('--n_train_data', type=str, default='', help='training data for source domain 2')
    parser.add_argument('--m_test_data', type=str, default='', help='testing data for source domain 1')
    parser.add_argument('--n_test_data', type=str, default='', help='testing data for source domain 2')
    parser.add_argument('--lambda', type=float, default=1, help='parameter for GRL (default: 1)')
    parser.add_argument('--gamma1', type=float, default=1, help='parameter source loss (default: 1)')
    parser.add_argument('--gamma2', type=float, default=0.01, help='parameter diff loss (default: 0.01)')
    parser.add_argument('--gamma3', type=float, default=0.01, help='parameter discriminator loss (default: 0.01)')
    parser.add_argument('--gamma4', type=float, default=1, help='parameter shared loss (default: 1)')
    parser.add_argument('--pad_len', type=int, default=1000, help='padding length for training data')
    parser.add_argument('--freeze_n_astnn', action='store_true', default=False, help='freeze ASTNN for domain n')

    return parser


def blocks_to_index(df, w2v):
    vocab = w2v.vocab
    max_token = w2v.vectors.shape[0]

    def tree_to_index(node):
        token = node.token
        result = [vocab[token].index if token in vocab else max_token]
        children = node.children
        for child in children:
            result.append(tree_to_index(child))
        return result

    def trans2seq(blocks):
        tree = []
        for b in blocks:
            btree = tree_to_index(b)
            tree.append(btree)
        return tree

    df.loc[:, 'blocks_seq'] = df['blocks'].apply(trans2seq)
    return df


def tokens_to_index(df, w2v):
    vocab = w2v.vocab
    max_token = w2v.vectors.shape[0]

    def trans2seq(token_list):
        result = [vocab[token].index if token in vocab else max_token for token in token_list]
        return result

    df.loc[:, 'tokens_seq'] = df['blocks'].apply(trans2seq)
    return df


def get_batch_data(dataset, batch_size, pad_len=None, expert=False, time=False, shuffle=True):

    if shuffle:
        # shuffle data
        dataset = dataset.sample(frac=1)
        dataset = dataset.reset_index(drop=True)

    if expert or time:
        return get_batch_data_expert_time(dataset, batch_size, time)

    if 'positive' in list(dataset) and 'negative' in list(dataset):
        return get_batch_data_triplet(dataset, batch_size)

    # check if sample pad_len is needed
    if pad_len is not None and len(dataset) > pad_len:
        dataset = dataset.sample(n=pad_len, random_state=1)
        dataset.reset_index(drop=True, inplace=True)

    # get batch data
    start_idx, end_idx = 0, batch_size
    batched_data = []
    while start_idx < len(dataset):
        cur_batch = dataset.iloc[start_idx: end_idx]

        inputs, labels = [], []
        for _, item in cur_batch.iterrows():  # DataFrame(columns=['sid', 'code', 'label'])
            inputs.append(item['code'])
            labels.append(item['label'])

        batch_tuple = (inputs, torch.FloatTensor(labels).unsqueeze(1))
        batched_data.append(batch_tuple)

        start_idx = end_idx
        end_idx += batch_size
    return batched_data, min(end_idx, len(dataset))


def get_batch_data_expert_time(dataset, batch_size, time=False):
    # get batch data
    start_idx, end_idx = 0, batch_size
    batched_data = []

    if time:
        col = 'time'
    else:
        col = 'expert'

    while start_idx < len(dataset):
        cur_batch = dataset.iloc[start_idx: end_idx]

        inputs, labels, experts = [], [], []
        for _, item in cur_batch.iterrows():  # DataFrame(columns=['sid', 'code', 'expert'/'time', 'label'])
            inputs.append(item['code'])
            labels.append(item['label'])
            experts.append(item[col])

        batch_tuple = ((inputs, experts), torch.FloatTensor(labels).unsqueeze(1))
        batched_data.append(batch_tuple)

        start_idx = end_idx
        end_idx += batch_size
    return batched_data, min(end_idx, len(dataset))


def get_batch_data_triplet(dataset, batch_size):
    # get batch data
    start_idx, end_idx = 0, batch_size
    batched_data = []
    while start_idx < len(dataset):
        cur_batch = dataset.iloc[start_idx: end_idx]

        inputs, labels = [], []
        po, ne = [], []

        for _, item in cur_batch.iterrows():  # DataFrame(columns=['sid', 'code', 'positive', 'negative', 'label'])
            inputs.append(item['code'])
            labels.append(item['label'])

            po.append(dataset.loc[dataset['sid'] == item['positive'], 'code'].values.tolist()[0])
            ne.append(dataset.loc[dataset['sid'] == item['negative'], 'code'].values.tolist()[0])

        batch_tuple = (inputs, po, ne, torch.FloatTensor(labels).unsqueeze(1))
        batched_data.append(batch_tuple)

        start_idx = end_idx
        end_idx += batch_size
    return batched_data, min(end_idx, len(dataset))


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cal_metrics(pred_list, label_list):
    pred, actual = np.array(pred_list).astype(int), np.array(label_list).astype(int)

    acc = accuracy_score(actual, pred)
    prec = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    f1 = f1_score(actual, pred)
    auc = roc_auc_score(actual, pred)
    cm = confusion_matrix(actual, pred)

    return acc, prec, recall, f1, auc, cm


def save_result(df, row, pred_list, label_list, digits=4):
    acc, prec, recall, f1, auc, cm = cal_metrics(pred_list, label_list)
    df.loc[row, 'Accuracy'] = round(acc, digits)
    df.loc[row, 'F1_score'] = round(f1, digits)
    df.loc[row, 'Precision'] = round(prec, digits)
    df.loc[row, 'Recall'] = round(recall, digits)
    df.loc[row, 'AUC'] = round(auc, digits)
    df.loc[row, 'Confusion_matrix'] = cm

    print("Confusion Matrix for {}: ".format(row))
    print(cm)

    print("Accuracy: ", acc)
    print("f1_score: ", f1)
    print("Precision: ", prec)
    print("Recall: ", recall)
    print("AUC: ", auc)

    return df


def prepare_input_temporal(df, w2v, train_s, test_s, mins, args, time=False, expert=False):
    print("\n-------------- Prepare [Temporal] input {} --------------".format(mins))

    MAX_STEP = 1500
    train_df = pd.DataFrame(columns=['sid', 'code', 'label'])
    test_df = pd.DataFrame(columns=['sid', 'code', 'label'])
    train_len, test_len = 0, 0
    max_length, min_length = 0, 1000

    grouped_data = df.groupby(['sid', 'semester', 'label'])
    for stu_info, stu_data in grouped_data:
        cur_student, cur_semester, cur_label = stu_info

        if cur_student in train_s or test_s:
            stu_data.index = range(stu_data.shape[0])  # reindex

            if expert:
                # read expert features
                blocks_seq = stu_data['expert'].values.tolist()
            else:
                stu_data = blocks_to_index(stu_data, w2v)
                blocks_seq = stu_data['blocks_seq'].values.tolist()
            # print (str(student) + ": " + str(len(value)))

            tmpx = []
            tmpy = int(cur_label)
            tmpt, prev_t = [], 0
            for i, stu_line in stu_data.iterrows():
                t = stu_line['time']
                if mins > 0 and t > mins * 60:
                    break
                else:
                    tmpx.append(blocks_seq[i])
                    tmpt.append(t-prev_t)
                    prev_t = t

            tmpx = tmpx[-MAX_STEP:]
            tmpt = tmpt[-MAX_STEP:]

            to_add_row = {'sid': cur_student, 'code': tmpx, 'label': tmpy}
            if time:
                # add time intervals
                to_add_row['time'] = tmpt

            if cur_student in train_s:
                train_df = train_df.append(to_add_row, ignore_index=True)
                train_len += 1

            elif cur_student in test_s:
                test_df = test_df.append(to_add_row, ignore_index=True)
                test_len += 1
            else:
                pass

            max_length = max(max_length, len(tmpx))
            min_length = min(min_length, len(tmpx))

    print("max length: ", max_length)
    print("min length: ", min_length)

    args.train_data = train_df
    args.test_data = test_df

    return train_len, test_len, args
