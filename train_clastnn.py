import time
import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.model_selection import KFold
from cl_trainer import CrossLingualTrainer
from utils.util import config_parser, blocks_to_index, save_result


def prepare_input(df, w2v, train_s, test_s, switch, args):
    print("\n-------------- Prepare input --------------")

    train_df = pd.DataFrame(columns=['sid', 'code', 'label'])
    test_df = pd.DataFrame(columns=['sid', 'code', 'label'])
    train_len, test_len = 0, 0

    blocks_df = blocks_to_index(df, w2v)
    for row_idx, row in blocks_df.iterrows():
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

    print("Data is ready for the domain [{}]".format(switch))
    print("training size: {}, testing size: {}".format(train_len, test_len))

    if switch == 'm':
        args.m_train_data = train_df
        args.m_test_data = test_df
    elif switch == 'n':
        args.n_train_data = train_df
        args.n_test_data = test_df
    else:
        raise ValueError('invalid input for parameter switch')

    return args


def run(command=None):
    if command is not None:
        args = config_parser().parse_args(command.split())
    else:
        args = config_parser().parse_args()

    # --------------------------------------- main code starts here --------------------------------------- #
    # load word2vec
    args.language = ''.join([args.m, args.n])
    word2vec = Word2Vec.load('embeddings/w2v_{}_{}'.format(args.language, args.embedding_dim)).wv
    max_tokens = word2vec.vectors.shape[0]
    embeddings = np.zeros((max_tokens + 1, args.embedding_dim), dtype="float32")
    embeddings[:max_tokens] = word2vec.vectors
    args.pretrained_embedding, args.vocab_size = embeddings, max_tokens + 1

    # load data
    langs = ['Snap', 'Java']
    m_data = pd.read_pickle('data/Snap/parsed_prob.pkl')
    n_data = pd.read_pickle('data/Java/parsed_prob.pkl')

    semester = m_data['semester'].unique().tolist()
    semester.sort(key=lambda x: (-int(x[-1]), x[0]), reverse=True)
    kf = KFold(n_splits=len(semester) - 1, shuffle=True)

    # result csv
    index_list = semester[1:] + langs
    res = pd.DataFrame(index=[s + '_shared' for s in index_list] + index_list,
                       columns=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1_score', 'Confusion_matrix'])

    post = 'rel'
    if 'freeze' in command:
        post = 'fix'
        if 'n_astnn' in command:
            post = 'fixN'
    file_info = '{}_{}_b{}lr{}_g2-{}_g3-{}_g4-{}'.format(args.name, post, args.batch_size, args.lr,
                                                         args.gamma2, args.gamma3, args.gamma4)
    res_file_name = 'result/res_{}.csv'.format(file_info)
    pred_file_name = 'result/pred_{}.csv'.format(file_info)

    # semester-based cv
    m_true, m_pred = [], []
    n_true, n_pred = [], []
    m_pred_shared, n_pred_shared = [], []
    for test_idx, (n_train, n_test) in zip(range(1, len(semester)), kf.split(n_data)):
        test_semester = semester[test_idx]
        train_semester = semester[:test_idx]
        print("\ntest semester:", test_semester, "train semester:", train_semester)

        data = m_data.loc[m_data['semester'].isin(train_semester + [test_semester]),]
        fold_m_train = data.loc[data['semester'].isin(train_semester), 'sid'].values.tolist()
        fold_m_test = data.loc[data['semester'].isin([test_semester]), 'sid'].values.tolist()

        fold_n_train = n_data.loc[n_train, 'sid'].values.tolist()
        fold_n_test = n_data.loc[n_test, 'sid'].values.tolist()

        args.pad_len = min(len(fold_m_train), len(fold_n_train))
        args = prepare_input(m_data, word2vec, fold_m_train, fold_m_test, 'm', args)
        args = prepare_input(n_data, word2vec, fold_n_train, fold_n_test, 'n', args)

        # train and retrieve results
        trainer = CrossLingualTrainer(args)
        m_fold_preds, m_fold_preds_shared, m_fold_trues, n_fold_preds, n_fold_preds_shared, n_fold_trues = \
            trainer.cross_lang_run(info=test_semester)
        res = save_result(res, test_semester, m_fold_preds, m_fold_trues)
        res = save_result(res, test_semester+'_shared', m_fold_preds_shared, m_fold_trues)
        res.to_csv(res_file_name)       # save file for each edit

        m_true.extend(m_fold_trues)
        m_pred.extend(m_fold_preds)
        m_pred_shared.extend(m_fold_preds_shared)

        n_true.extend(n_fold_trues)
        n_pred.extend(n_fold_preds)
        n_pred_shared.extend(n_fold_preds_shared)

        # calculate metrics
        for cur_lang, (cur_pred, cur_true) in zip(langs, [(m_pred, m_true), (n_pred, n_true)]):
            res = save_result(res, cur_lang, cur_pred, cur_true)
            res.to_csv(res_file_name)       # save file for each edit

        for cur_lang, (cur_pred, cur_true) in zip(langs, [(m_pred_shared, m_true), (n_pred_shared, n_true)]):
            res = save_result(res, cur_lang+'_shared', cur_pred, cur_true)
            res.to_csv(res_file_name)       # save file for each edit

    # save predictions
    pred_info = pd.DataFrame()
    pred_info.loc[:, 'Snap_private'] = m_pred
    pred_info.loc[:, 'Snap_shared'] = m_pred_shared
    pred_info.loc[:, 'Snap_label'] = m_true
    pred_info.to_csv(pred_file_name, index=False)


if __name__ == '__main__':
    command = '--name CrossLing --batch 20 --lr 0.1 --gamma2 0.1 --gamma3 1.0 --gamma4 1.0 --epochs 50 ' \
              '--load_pretrain'
    run(command)
