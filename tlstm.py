import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


'''
ref link: https://github.com/granmirupa/tLSTM/tree/a144d6905012ce1d17b72465c8d6a187e41b2c4f
'''


class TLSTM(nn.Module):
    def __init__(self, input_size, batch_size, hidden_size=64, num_labels=1):
        super(TLSTM, self).__init__()

        self.batch_first = True
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_labels = num_labels

        # lstm weights for previous labels
        self.weight_fm = nn.Linear(self.hidden_size, self.hidden_size)
        self.weight_im = nn.Linear(self.hidden_size, self.hidden_size)
        self.weight_cm = nn.Linear(self.hidden_size, self.hidden_size)
        self.weight_om = nn.Linear(self.hidden_size, self.hidden_size)

        # lstm weights for features
        self.weight_fx = nn.Linear(self.input_size, self.hidden_size)
        self.weight_ix = nn.Linear(self.input_size, self.hidden_size)
        self.weight_cx = nn.Linear(self.input_size, self.hidden_size)
        self.weight_ox = nn.Linear(self.input_size, self.hidden_size)

        # lstm weights for time features
        self.weight_tm = nn.Linear(self.hidden_size, self.hidden_size)

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.num_labels),
            nn.Sigmoid()
        )

    def forward(self, x):
        def elapsed_time(time):
            e = torch.exp(torch.Tensor([1]))
            denominator = torch.log(e + time)
            discounted_time = torch.div(1, denominator)

            return discounted_time

        def recurrence(input, time, hx, cx):
            """Recurrence helper."""

            # Previous state
            cx_short = F.tanh(self.weight_tm(cx))
            cx_long = cx - cx_short
            cx_new = cx_long + elapsed_time(time) * cx_short

            # Gates
            ingate = F.sigmoid(self.weight_ix(input) + self.weight_im(hx))
            forgetgate = F.sigmoid(self.weight_fx(input) + self.weight_fm(hx))
            outgate = F.sigmoid(self.weight_ox(input) + self.weight_om(hx))
            cellgate = F.tanh(self.weight_cx(input) + self.weight_cm(hx))

            cy = (forgetgate * cx_new) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # n_b x hidden_dim

            # hy = self.decoder(hy)

            return hy, cy

        feature_x, feature_t = x
        self.batch_size = len(feature_x)

        if not torch.is_tensor(feature_x) or not torch.is_tensor(feature_t):
            # Ye's note: pre-pad st vectors to the same length of max_len
            lens = [len(item) for item in feature_x]
            max_len = max(lens)
            seq_x, seq_t, start, end = [], [], 0, 0
            for i in range(self.batch_size):
                pad_len = max_len-lens[i]
                if pad_len > 0:
                    seq_x.append(Variable(torch.zeros(pad_len, len(feature_x[0][0]))))
                    seq_t.append(Variable(torch.zeros(pad_len)))
                seq_x.append(torch.FloatTensor(feature_x[i]))
                seq_t.append(torch.FloatTensor(feature_t[i]))

            encodes = torch.cat(seq_x)
            feature = encodes.view(self.batch_size, max_len, -1)  # (b, max_len, feature_dim)

            encodes = torch.cat(seq_t)
            time = encodes.view(self.batch_size, max_len, -1)  # (b, max_len, 1)

        else:
            feature, time = x

        if self.batch_first:
            feature = feature.transpose(0, 1)
            time = time.transpose(0, 1)

        output = []
        steps = range(feature.size(0))
        hidden, cell = self.init_hidden_()

        # print input
        for i in steps:
            hidden, cell = recurrence(feature[i], time[i], hidden, cell)
            output.append(hidden)

        output = torch.cat(output, 0).view(feature.size(0), feature.size(1), self.hidden_size)

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden

    def init_hidden_(self):
        h0 = Variable(torch.zeros(self.batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(self.batch_size, self.hidden_size))
        return h0, c0


class NormalTLSTM(nn.Module):
    """
    Normal TLSTM classifier
    """
    def __init__(self, args):
        super(NormalTLSTM, self).__init__()
        self.input_size = 7
        self.hidden_size = 64
        self.batch_size = args['batch_size']
        self.num_labels = args['label_size']

        self.tlstm = TLSTM(self.input_size, self.batch_size, self.hidden_size, self.num_labels)
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.num_labels),
            nn.Sigmoid()
        )

    def forward(self, x):
        out, _ = self.tlstm(x)
        y = self.decoder(out[:, -1, :])
        return y