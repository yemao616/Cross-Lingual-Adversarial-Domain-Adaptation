import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score, precision_score
from trainer import Trainer
from utils.util import config_parser, prepare_input_temporal, save_result


def run(command=None, mins=2, use_semester=True):
    if command is not None:
        args = config_parser().parse_args(command.split())
    else:
        args = config_parser().parse_args()

    # ---- load data
    all_data = pd.read_pickle('./data/{}/parsed_programs.pkl'.format(args.language))
    semester = all_data['semester'].unique().tolist()
    semester.sort(key=lambda x: (-int(x[-1]), x[0]), reverse=True)
    res = pd.DataFrame(index=semester + ['overall'],
                       columns=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1_score', 'Confusion_matrix'])
    res_file = 'result/res_{}m_{}.csv'.format(mins, args.name)

    all_true, all_pred = [], []
    for test_idx in range(1, len(semester)):
        test_semester = semester[test_idx]
        train_semester = semester[:test_idx]
        print("\ntest semester:", test_semester, "train semester:", train_semester)

        data = all_data.loc[all_data['semester'].isin(train_semester + [test_semester]), ]
        train_sid = np.array(data.loc[data['semester'].isin(train_semester), 'sid'].unique().tolist())
        test_sid = data.loc[data['semester'].isin([test_semester]), 'sid'].unique().tolist()

        info_list = [test_semester, str(mins)]

        if use_semester:
            # ---- load semester-specific w2v
            word2vec = Word2Vec.load(
                './embeddings/w2v_{}_{}_{}'.format(args.language, args.embedding_dim, test_semester)).wv
            info_list.append('semester')

        else:
            # ---- load SnapJava w2v
            word2vec = Word2Vec.load('./embeddings/w2v_{}_{}'.format(args.language, args.embedding_dim)).wv

        max_tokens = word2vec.vectors.shape[0]
        embeddings = np.zeros((max_tokens + 1, args.embedding_dim), dtype="float32")
        embeddings[:max_tokens] = word2vec.vectors
        args.pretrained_embedding, args.vocab_size = embeddings, max_tokens + 1

        train_num, test_num, args = prepare_input_temporal(df=data, train_s=train_sid, test_s=test_sid, w2v=word2vec,
                                                           mins=mins, args=args, time=True)
        print("training size:", train_num, "testing size:", test_num)

        trainer = Trainer(args)
        pred, actual = trainer.run(info='_'.join(info_list))

        # save best result for each semester
        res = save_result(res, test_semester, pred, actual)
        res.to_csv(res_file)

        # append predictions and labels
        all_true.extend(actual)
        all_pred.extend(pred)

    # save overall result
    res = save_result(res, 'overall', all_pred, all_true)
    res.to_csv(res_file)


if __name__ == '__main__':
    cur_command = '--name TemporalCrossLing --lang Snap --time --batch 20 --epochs 50'
    for cur_min in [2, 4, 6, 8, 10]:
        run(command=cur_command, mins=cur_min, use_semester=True)
