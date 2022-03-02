import os
import pandas as pd
from gensim.models.word2vec import Word2Vec


class Pipeline():
    def __init__(self, language, seqential=False):
        if language == 'Java':
            from utils.utils_Java import parsing_trees as program_parser
            from utils.utils_Java import get_sequences as get_embedding_sequences
            from utils.utils_Java import get_blocks as get_tree_blocks

        elif language == 'Snap':
            from utils.utils_Snap import parsing_trees as program_parser
            from utils.utils_Snap import get_sequences as get_embedding_sequences
            from utils.utils_Snap import get_blocks as get_tree_blocks
        else:
            raise ValueError('unsupported language: {}, only support Java and Snap'.format(language))

        self.language = language
        self.sequential = seqential
        self.program_parser = program_parser
        self.get_embedding_sequences = get_embedding_sequences
        self.get_tree_blocks = get_tree_blocks
        self.source = None
        self.output_file = None

    def parser(self):
        if self.sequential:
            filename = 'programs'
        else:
            filename = 'prob'
        input_file = 'data/{}/raw_{}.pkl'.format(self.language, filename)
        output_file = 'data/{}/parsed_{}.pkl'.format(self.language, filename)
        print('--> parsing source code from {}...'.format(input_file))
        source = pd.read_pickle(input_file)

        if self.sequential:
            source.columns = ['sid', 'time', 'semester', 'code_id', 'raw_code', 'label']
            source.loc[:, 'code'] = source['raw_code'].apply(self.program_parser)
            source.loc[:, 'label'] = [1 if each == 'U' else 0 for each in source.label]
        else:
            source.columns = ['sid', 'semester', 'code_id', 'code', 'label']
            source.loc[:, 'code'] = source['code'].apply(self.program_parser)

        source = source.reset_index(drop=True)
        self.source = source
        self.output_file = output_file
        return source

    def get_sequences(self):
        print('--> generating pre-order sequences...')
        def trans_to_sequences(ast):
            sequence = []
            self.get_embedding_sequences(ast, sequence)
            return sequence

        self.source.loc[:, 'tokens'] = self.source['code'].apply(trans_to_sequences)

    def get_blocks(self):
        print('--> generating statement blocks...')
        def trans2seq(r):
            blocks = []
            self.get_tree_blocks(r, blocks)
            return blocks

        self.source.loc[:, 'blocks'] = self.source['code'].apply(trans2seq)

    # may not be used for cross-lingual framework
    def train_semester_embeddings(self, size):
        semester = self.source['semester'].unique().tolist()
        semester.sort(key=lambda x: (-int(x[-1]), x[0]), reverse=True)
        for test_idx in range(1, len(semester)):
            test_semester = semester[test_idx]
            train_semester = semester[:test_idx]
            corpus = self.source.loc[self.source['semester'].isin(train_semester), 'tokens']

            print('--> training word embedding corpus to test semester {}'.format(test_semester))
            print(str(train_semester), len(corpus))
            w2v = Word2Vec(corpus, size=size, workers=16, sg=1, max_final_vocab=3000)
            if self.sequential:
                w2v_file = 'embeddings/temporal/w2v_{}_{}_{}'.format(self.language, size, test_semester)
            else:
                w2v_file = 'embeddings/w2v_{}_{}_{}'.format(self.language, size, test_semester)
            print('save w2v file to {}\n'.format(w2v_file))
            w2v.save(w2v_file)

    def train_embedding(self, size):
        corpus = self.source['tokens'].values.tolist()
        w2v = Word2Vec(corpus, size=size, workers=16, sg=1, max_final_vocab=3000, min_count=2, iter=10)
        if self.sequential:
            w2v_file = 'embeddings/w2v_{}_{}'.format(self.language, size)
        else:
            w2v_file = 'embeddings/temporal/w2v_{}_{}'.format(self.language, size)
        print('save w2v file to {}\n'.format(w2v_file))
        w2v.save(w2v_file)

    def run(self, size=128):
        self.parser()
        self.get_sequences()
        self.get_blocks()
        self.train_embedding(size)

        if self.language == 'Java':
            from utils.utils_Java import pad_blocks
            self.source.loc[:, 'blocks'] = self.source['blocks'].apply(pad_blocks)
        else:
            self.train_semester_embeddings(size)

        print("save parsed code to {}\n".format(self.output_file))
        self.source.to_pickle(self.output_file)
        return self.source


def cross_lingual_embeddigngs(source1, source2, size=128, sequential=False):
    corpus_1 = source1['tokens'].values.tolist()
    corpus_2 = source2['tokens'].values.tolist()
    corpus = corpus_1 + corpus_2
    w2v = Word2Vec(corpus, size=size, workers=16, sg=1, max_final_vocab=3000, min_count=4, iter=10)
    w2v_file = 'w2v_SnapJava_{}'.format(size)
    full_w2v_file = 'embeddings/temporal/' if sequential else 'embeddings/'
    full_w2v_file += w2v_file
    print('save w2v file to {}\n'.format(full_w2v_file))
    w2v.save(full_w2v_file)


embed_size = 64
ppl1 = Pipeline('Snap', True)
s1 = ppl1.run(embed_size)

ppl2 = Pipeline('Java')
s2 = ppl2.run(embed_size)

cross_lingual_embeddigngs(s1, s2, embed_size, True)
