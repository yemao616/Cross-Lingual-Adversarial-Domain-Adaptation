import torch
import torch.nn as nn
from torch.autograd import Variable
from tlstm import TLSTM
from astnn import ASTNN


class CrossLing(nn.Module):
    """
    Cross-Language ASTNN Model
    """

    def __init__(self, args):
        super(CrossLing, self).__init__()
        self.with_target_domain = args['with_target_domain']

        self.mASTNN = ASTNN(args)
        self.nASTNN = ASTNN(args)
        self.sharedASTNN = ASTNN(args)

        self.grl = GradientReversal(args['lambda'])
        self.discriminator = Classifier(args['hidden_dim'], args['domain_size'])
        self.shared2label = Classifier(args['hidden_dim'], args['label_size'])
        self.m2label = Classifier(args['hidden_dim'], args['label_size'], True)
        self.n2label = Classifier(args['hidden_dim'], args['label_size'], True)

    def forward(self, m, n, m_share=None, n_share=None, t=None):
        # private ASTNN
        m_hidden = self.mASTNN(m)
        n_hidden = self.nASTNN(n)

        # shared ASTNN
        if m_share:
            m_shared_hidden = self.sharedASTNN(m_share)
        else:
            m_shared_hidden = self.sharedASTNN(m)

        if n_share:
            n_shared_hidden = self.sharedASTNN(n_share)
        else:
            n_shared_hidden = self.sharedASTNN(n)

        # private classifier
        m_prediction = self.m2label(torch.cat([m_hidden, m_shared_hidden], dim=1))
        n_prediction = self.n2label(torch.cat([n_hidden, n_shared_hidden], dim=1))

        # Discriminator
        m_domain = self.discriminator(self.grl(m_shared_hidden))
        n_domain = self.discriminator(self.grl(n_shared_hidden))

        # shared classifier
        m_shared_prediction = self.shared2label(m_shared_hidden)
        n_shared_prediction = self.shared2label(n_shared_hidden)

        m_output = [m_domain, m_prediction, m_shared_prediction, m_hidden, m_shared_hidden]
        n_output = [n_domain, n_prediction, n_shared_prediction, n_hidden, n_shared_hidden]

        if self.with_target_domain and t:
            t_shared_hidden = self.sharedASTNN(t)                       # pass shared ASTNN
            t_domain = self.discriminator(self.grl(t_shared_hidden))    # pass discriminator
            t_output = t_domain

            return m_output, n_output, t_output

        return m_output, n_output

    def forward_predict(self, inputs, switch):
        """
        cl-astnn function to generate input for different switch (domain)
        """
        if switch == 'm':
            m_hidden = self.mASTNN(inputs)
            m_shared_hidden = self.sharedASTNN(inputs)
            output = self.m2label(torch.cat([m_hidden, m_shared_hidden], dim=1))

        elif switch == 'n':
            n_hidden = self.nASTNN(inputs)
            n_shared_hidden = self.sharedASTNN(inputs)
            output = self.n2label(torch.cat([n_hidden, n_shared_hidden], dim=1))

        elif switch == 't':
            t_shared_hidden = self.sharedASTNN(inputs)
            output = self.shared2label(t_shared_hidden)
        else:
            raise ValueError('switch must be one of (m, n, t)')

        return output

    def forward_predict_tsne(self, inputs, switch='m'):
        if switch == 'm':
            hidden = self.mASTNN(inputs)
            shared_hidden = self.sharedASTNN(inputs)
            output = self.m2label(torch.cat([hidden, shared_hidden], dim=1))

        else:
            hidden = self.nASTNN(inputs)
            shared_hidden = self.sharedASTNN(inputs)
            output = self.n2label(torch.cat([hidden, shared_hidden], dim=1))

        return output, hidden, shared_hidden


class GradientReversal(nn.Module):
    """
    Gradient Reversal Layer

    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)


    Ref: Y. Ganin et al., Domain-adversarial training of neural networks (2016)
    Link: https://jmlr.csail.mit.edu/papers/volume17/15-239/15-239.pdf
    """

    def __init__(self, lambda_):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return x

    def backward(self, x):
        return -self.lambd * x


class Classifier(nn.Module):
    """
    Classification Layer
    """

    def __init__(self, hidden_dim, output_dim, cat_hidden=False, bidirection=True):
        super(Classifier, self).__init__()
        if bidirection:
            hidden_dim = 2 * hidden_dim

        if cat_hidden:
            hidden_dim = 2 * hidden_dim

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = torch.sigmoid(self.fc(x))
        # tmp=self.fc2(tmp)
        # tmp=self.fc3(tmp)
        # tmp = F.log_softmax(tmp)
        return out


class TemporalCrossLing(nn.Module):
    """
    Temporal CrossLing Model
    """

    def __init__(self, args):
        super(TemporalCrossLing, self).__init__()
        self.astnn = ASTNN(args)
        self.Gastnn = ASTNN(args)
        self.gpu = args['use_gpu']
        self.input_size = args['hidden_dim'] * 4
        self.pad_size = args['hidden_dim'] * 2
        self.batch_size = args['batch_size']
        self.label_size = args['label_size']
        self.time = args['time']
        self.hidden_size = 64
        self.num_layers = 1
        if self.time:
            self.lstm = TLSTM(self.input_size, self.batch_size, self.hidden_size, self.label_size)
        else:
            self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.label_size)

    def pad_zeros(self, num):
        zeros = Variable(torch.zeros(num, self.pad_size))
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
        if self.time:
            # using TLSTM
            feature, time = x
            x = feature
        else:
            time = None

        lens = [len(item) for item in x]
        max_len = max(lens)
        self.batch_size = len(x)

        batch_ast = []
        Gbatch_ast = []
        batch_t = []
        for i in range(self.batch_size):  # each student automatically form a batch for astnn
            pad_len = max(max_len - lens[i], 0)
            if pad_len > 0:
                cur_pad = self.pad_zeros(pad_len)
                batch_ast.append(cur_pad)
                Gbatch_ast.append(cur_pad)
                batch_t.append(Variable(torch.zeros(pad_len)))

            cur_encode = self.astnn(x[i])
            batch_ast.append(cur_encode)

            cur_Gencode = self.Gastnn(x[i])
            Gbatch_ast.append(cur_Gencode)

            if self.time:
                batch_t.append(torch.FloatTensor(time[i]))

        encodes = torch.cat(batch_ast)
        encodes = encodes.view(self.batch_size, max_len, -1)

        Gencodes = torch.cat(Gbatch_ast)
        Gencodes = Gencodes.view(self.batch_size, max_len, -1)

        if self.time:
            encodes_t = torch.cat(batch_t)
            time = encodes_t.view(self.batch_size, max_len, -1)  # (b, max_len, 1)

        all_encodes = torch.cat([encodes, Gencodes], dim=2)

        if self.gpu:
            all_encodes = Variable(all_encodes).cuda()
            time = time.cuda()

        if self.time:
            out, _ = self.lstm((all_encodes, time))
        else:
            # Set initial hidden and cell states
            h0, c0 = self.init_hidden()

            # Forward propagate LSTM
            out, _ = self.lstm(all_encodes, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        return torch.sigmoid(out)

