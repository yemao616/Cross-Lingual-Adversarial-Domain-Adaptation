import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from tlstm import TLSTM


class BatchTreeEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encode_dim, batch_size, use_gpu, pretrained_weight=None):
        super(BatchTreeEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encode_dim = encode_dim
        self.W_c = nn.Linear(embedding_dim, encode_dim)
        self.W_l = nn.Linear(encode_dim, encode_dim)
        self.W_r = nn.Linear(encode_dim, encode_dim)
        self.activation = F.relu
        self.stop = -1
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.node_list = []
        self.th = torch.cuda if use_gpu else torch
        self.batch_node = None
        # pretrained  embedding
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            # self.embedding.weight.requires_grad = False

    def create_tensor(self, tensor):
        if self.use_gpu:
            return tensor.cuda()
        return tensor

    def traverse_mul(self, node, batch_index):
        size = len(node)
        if not size:
            return None
        batch_current = self.create_tensor(Variable(torch.zeros(size, self.encode_dim)))

        index, children_index = [], []
        current_node, children = [], []
        for i in range(size):
            if node[i][0] != -1:
                index.append(i)
                current_node.append(node[i][0])
                temp = node[i][1:]
                c_num = len(temp)
                for j in range(c_num):
                    if temp[j][0] != -1:
                        if len(children_index) <= j:
                            children_index.append([i])
                            children.append([temp[j]])
                        else:
                            children_index[j].append(i)
                            children[j].append(temp[j])
            else:
                batch_index[i] = -1

        batch_current = self.W_c(batch_current.index_copy(0, Variable(self.th.LongTensor(index)),
                                                          self.embedding(Variable(self.th.LongTensor(current_node)))))

        for c in range(len(children)):
            zeros = self.create_tensor(Variable(torch.zeros(size, self.encode_dim)))
            batch_children_index = [batch_index[i] for i in children_index[c]]
            tree = self.traverse_mul(children[c], batch_children_index)
            if tree is not None:
                batch_current += zeros.index_copy(0, Variable(self.th.LongTensor(children_index[c])), tree)
        # batch_current = F.tanh(batch_current)
        batch_index = [i for i in batch_index if i != -1]
        b_in = Variable(self.th.LongTensor(batch_index))
        self.node_list.append(self.batch_node.index_copy(0, b_in, batch_current))
        return batch_current

    def forward(self, x, bs):
        self.batch_size = bs
        self.batch_node = self.create_tensor(Variable(torch.zeros(self.batch_size, self.encode_dim)))
        self.node_list = []
        self.traverse_mul(x, list(range(self.batch_size)))
        self.node_list = torch.stack(self.node_list)
        return torch.max(self.node_list, 0)[0]


class ASTNN(nn.Module):
    """
    AST-based NN Model
    """
    def __init__(self, args):

        super(ASTNN, self).__init__()
        self.num_layers = 1
        self.gpu = args['use_gpu']
        self.batch_size = args['batch_size']
        self.vocab_size = args['vocab_size']
        self.embedding_dim = args['embedding_dim']
        self.encode_dim = args['encode_dim']
        self.hidden_dim = args['hidden_dim']
        self.bn = nn.BatchNorm1d(args['hidden_dim']*2)
        self.stop = [self.vocab_size - 1]

        # class "BatchTreeEncoder"
        self.encoder = BatchTreeEncoder(self.vocab_size, self.embedding_dim, self.encode_dim,
                                        self.batch_size, self.gpu, args['pretrained_embedding'])

        # gru
        self.bigru = nn.GRU(self.encode_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=True,
                            batch_first=True)


        # hidden
        self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(0.2)

    def init_hidden(self):
        if self.gpu is True:
            if isinstance(self.bigru, nn.LSTM):
                h0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
                c0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
                return h0, c0
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim)).cuda()
        else:
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim))

    def get_zeros(self, num):
        zeros = Variable(torch.zeros(num, self.encode_dim))
        if self.gpu:
            return zeros.cuda()
        return zeros

    def forward(self, x):
        lens = [len(item) for item in x]
        max_len = max(lens)

        encodes = []
        self.batch_size = len(x)
        for i in range(self.batch_size):
            for j in range(lens[i]):
                encodes.append(x[i][j])

        encodes = self.encoder(encodes, sum(lens))

        # Ye's note: pre-pad st vectors to the same length of max_len
        seq, start, end = [], 0, 0
        for i in range(self.batch_size):
            end += lens[i]
            if max_len-lens[i]:
                seq.append(self.get_zeros(max_len-lens[i]))
            seq.append(encodes[start:end])
            start = end
        encodes = torch.cat(seq)
        encodes = encodes.view(self.batch_size, max_len, -1)  # (b, max_len, encode_dim)

        # gru
        gru_out, hidden = self.bigru(encodes, self.init_hidden())  # (b, max_len, hidden_dim *2)

        gru_out = torch.transpose(gru_out, 1, 2)  # (b, hidden_dim *2, max_len)

        # Ye's note: add batch norm
        gru_out = self.bn(gru_out)

        # pooling
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)  # (b, hidden_dim *2)
        # Ye's note: squeeze is to remove dimensions of size 1
        # gru_out = gru_out[:,-1]

        return gru_out


class NormalASTNN(nn.Module):
    """
    Normal ASTNN classifier
    """
    def __init__(self, args):
        super(NormalASTNN, self).__init__()
        self.astnn = ASTNN(args)
        self.hidden2label = nn.Linear(args['hidden_dim'] * 2, args['label_size'])

    def forward(self, x):
        hidden = self.astnn(x)
        y = torch.sigmoid(self.hidden2label(hidden))
        return y

    def forward_predict_tsne(self, x):
        hidden = self.astnn(x)
        output = torch.sigmoid(self.hidden2label(hidden))

        return output, hidden


class TripletASTNN(nn.Module):
    """
    Triplet Loss ASTNN

    Code ref link: https://github.com/andreasveit/triplet-network-pytorch
    """
    def __init__(self, args):
        super(TripletASTNN, self).__init__()
        self.alpha = args['alpha']
        self.astnn = ASTNN(args)
        self.hidden2label = nn.Linear(args['hidden_dim'] * 2, args['label_size'])

    def l2_norm(self, embed):
        norm = torch.norm(embed, p=2, dim=1).unsqueeze(1).expand_as(embed)
        normed_embed = embed.div(norm + 1e-5)
        normed_embed = normed_embed * self.alpha
        return norm, normed_embed

    def forward(self, x, y, z):
        embed_x = self.astnn(x)
        embed_y = self.astnn(y)
        embed_z = self.astnn(z)
        norm_x, normed_embed_x = self.l2_norm(embed_x)
        norm_y, normed_embed_y = self.l2_norm(embed_y)
        norm_z, normed_embed_z = self.l2_norm(embed_z)
        dist_a = F.pairwise_distance(normed_embed_x, normed_embed_y, 2)
        dist_b = F.pairwise_distance(normed_embed_x, normed_embed_z, 2)
        output = torch.sigmoid(self.hidden2label(embed_x))
        return dist_a, dist_b, normed_embed_x, normed_embed_y, normed_embed_z, output

    def forward_predict(self, inputs):
        embed_inputs = self.astnn(inputs)
        output = torch.sigmoid(self.hidden2label(embed_inputs))
        return output


class TemporalASTNN(nn.Module):
    """
    L-ASTNN Model
    """
    def __init__(self, args):
        super(TemporalASTNN, self).__init__()
        self.astnn = ASTNN(args)
        self.gpu = args['use_gpu']
        self.input_size = args['hidden_dim']*2
        self.batch_size = args['batch_size']
        self.label_size = args['label_size']
        self.hidden_size = 64
        self.num_layers = 1
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.label_size)

    def pad_zeros(self, num):
        zeros = Variable(torch.zeros(num, self.input_size))
        if self.gpu:
            return zeros.cuda()
        return zeros

    def init_hidden(self):
        if self.gpu is True:
            h0 = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size).cuda())
            c0 = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size).cuda())
        else:
            h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
            c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        return h0, c0

    def forward(self, x):
        lens = [len(item) for item in x]
        max_len = max(lens)
        self.batch_size = len(x)

        batch_ast = []
        for i in range(self.batch_size):    # each student automatically form a batch for astnn
            pad_len = max(max_len - lens[i], 0)
            if pad_len > 0:
                batch_ast.append(self.pad_zeros(pad_len))

            cur_encode = self.astnn(x[i])
            batch_ast.append(cur_encode)

        encodes = torch.cat(batch_ast)
        encodes = encodes.view(self.batch_size, max_len, -1)

        if self.gpu:
            encodes = Variable(encodes).cuda()

        # Set initial hidden and cell states
        h0, c0 = self.init_hidden()

        # Forward propagate LSTM
        out, _ = self.lstm(encodes, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        return torch.sigmoid(out)


class TLASTNN(nn.Module):
    """
    TLSTM-ASTNN Model
    """
    def __init__(self, args):
        super(TLASTNN, self).__init__()
        self.astnn = ASTNN(args)
        self.gpu = args['use_gpu']
        self.input_size = args['hidden_dim']*2
        self.batch_size = args['batch_size']
        self.label_size = args['label_size']
        self.hidden_size = 64
        self.num_layers = 1
        self.tlstm = TLSTM(self.input_size, self.batch_size, self.hidden_size, self.label_size)
        self.fc = nn.Linear(self.hidden_size, self.label_size)

    def forward(self, x):
        feature, time = x
        self.batch_size = len(feature)
        lens = [len(item) for item in feature]
        max_len = max(lens)

        batch_ast = []
        batch_t = []
        for i in range(self.batch_size):    # each student automatically form a batch for astnn
            pad_len = max(max_len - lens[i], 0)
            if pad_len > 0:
                batch_ast.append(Variable(torch.zeros(pad_len, self.input_size)))
                batch_t.append(Variable(torch.zeros(pad_len)))

            cur_encode = self.astnn(feature[i])
            batch_ast.append(cur_encode)

            batch_t.append(torch.FloatTensor(time[i]))

        encodes = torch.cat(batch_ast)
        encodes = encodes.view(self.batch_size, max_len, -1)

        encodes_t = torch.cat(batch_t)
        time = encodes_t.view(self.batch_size, max_len, -1)  # (b, max_len, 1)

        if self.gpu:
            encodes = Variable(encodes).cuda()
            time = time.cuda()

        out, _ = self.tlstm((encodes, time))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        return torch.sigmoid(out)

