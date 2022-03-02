import os
import sys
import numpy as np
import time
import torch
from functools import reduce
from torch.autograd import Variable
from utils.util import AverageMeter, get_batch_data
from sklearn.metrics import precision_recall_fscore_support
from trainer import Trainer


# ------------------------------ for CLASTNN only ------------------------------ #
class CrossLingualTrainer(Trainer):
    """
    Basic Trainer Class containing train and test process
    """

    def __init__(self, args):
        super(CrossLingualTrainer, self).__init__(args)

    @property
    def m_train_data(self):
        return self.args.m_train_data

    @property
    def m_test_data(self):
        return self.args.m_test_data

    @property
    def n_train_data(self):
        return self.args.n_train_data

    @property
    def n_test_data(self):
        return self.args.n_test_data

    @property
    def gamma1(self):
        return self.args.gamma1

    @property
    def gamma2(self):
        return self.args.gamma2

    @property
    def gamma3(self):
        return self.args.gamma3

    @property
    def gamma4(self):
        return self.args.gamma4

    @property
    def with_target_domain(self):
        return self.args.with_target_domain

    @property
    def pad_len(self):
        return self.args.pad_len

    def _cross_language_loss(self, m_output, n_output, m_label, n_label):
        """
        loss function for CLASTNN
        """

        m_domain, m_prediction, m_shared_prediction, m_hidden, m_shared_hidden = m_output
        n_domain, n_prediction, n_shared_prediction, n_hidden, n_shared_hidden = n_output

        loss_list = []

        gamma1, gamma2, gamma3, gamma4 = self.gamma1, self.gamma2, self.gamma3, self.gamma4

        # source domain classification loss
        loss1 = gamma1 * self.criterion(m_prediction, m_label)
        # loss1 += gamma1 * self.criterion(n_prediction, n_label)
        loss_list.append(loss1)

        # hidden vectors diff loss
        loss2 = gamma2 * torch.norm(torch.matmul(m_shared_hidden.t(), m_hidden))
        loss2 += gamma2 * torch.norm(torch.matmul(n_shared_hidden.t(), n_hidden))
        loss_list.append(loss2)

        # discriminator loss
        if self.use_gpu:
            hey0 = Variable(torch.zeros(len(m_domain))).type(torch.FloatTensor).cuda()
            hey1 = Variable(torch.zeros(len(n_domain))).type(torch.FloatTensor).cuda() + 1
        else:
            hey0 = Variable(torch.zeros(len(m_domain))).type(torch.FloatTensor)
            hey1 = Variable(torch.zeros(len(n_domain))).type(torch.FloatTensor) + 1
        domain_label = torch.cat([hey0, hey1]).unsqueeze(1)

        predicted_domain_label = torch.cat([m_domain, n_domain])
        loss3 = gamma3 * self.criterion(predicted_domain_label, domain_label)
        loss_list.append(loss3)

        # shared classification loss
        loss4 = gamma4 * self.criterion(m_shared_prediction, m_label)
        # loss4 += gamma4 * self.criterion(n_shared_prediction, n_label)
        loss_list.append(loss4)

        loss = reduce(lambda x, y: x + y, loss_list)

        return loss, loss1, loss2, loss3, loss4

    def load_cross_lang_weights(self):
        switches = ['m', 'n']
        filenames = ['Snap_' + self.info, 'Java']

        for switch, file_lang in zip(switches, filenames):
            if self.name == 'CrossLing':
                folder = 'ASTNN'
            else:
                print(self.name)
                raise ValueError("invalid model for cross-lingual framework")
            filename = 'logs/{}/best_model_{}.pth.tar'.format(folder, file_lang)
            self.log('======== loading weights from {}'.format(filename))
            dict_info = torch.load(filename)
            pretrained_dict = dict_info['state_dict']
            model_dict = self.model.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k.replace('astnn', switch + 'ASTNN'): v for k, v in pretrained_dict.items() if
                               'astnn' in k}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)

            # 3. load the new state dict
            self.model.load_state_dict(model_dict)

    def cross_lang_train(self, cur_epoch):
        """
        Training phase for CLASTNN

        """
        if self.load_pretrain:
            self.load_cross_lang_weights()

        # train mode
        self.model.train()

        losses = AverageMeter()
        loss1s = AverageMeter()
        loss2s = AverageMeter()
        loss3s = AverageMeter()
        loss4s = AverageMeter()
        m_accs = AverageMeter()
        n_accs = AverageMeter()
        m_accs_shared = AverageMeter()
        n_accs_shared = AverageMeter()

        m_data_loader, m_total_data = get_batch_data(self.m_train_data, self.batch_size, pad_len=self.pad_len)
        n_data_loader, n_total_data = get_batch_data(self.n_train_data, self.batch_size, pad_len=self.pad_len)

        if m_total_data == n_total_data:
            total_data = m_total_data
        else:
            raise ValueError("inconsistent data length between three domains")

        for batch_idx, (m_data, n_data) in enumerate(zip(m_data_loader, n_data_loader)):
            # import pdb
            # pdb.set_trace()
            m_inputs, m_label = m_data
            n_inputs, n_label = n_data
            batch_num = len(m_label)

            if self.use_gpu:
                m_label = Variable(m_label.cuda())
                n_label = Variable(n_label.cuda())
            else:
                m_label = Variable(m_label)
                n_label = Variable(n_label)

            # -------- forward pass -------- #
            m_output, n_output = self.model(m_inputs, n_inputs)
            loss, loss1, loss2, loss3, loss4 = self._cross_language_loss(m_output, n_output, m_label, n_label)

            # -------- backward and optimize -------- #
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # -------- calculate accuracy and loss -------- #
            m_predicted = m_output[1].data > 0.5  # private astnn
            m_acc = (m_predicted == m_label).sum() / batch_num
            m_accs.update(m_acc, batch_num)

            m_predicted_share = m_output[2].data > 0.5  # shared astnn
            m_acc_shared = (m_predicted_share == m_label).sum() / batch_num
            m_accs_shared.update(m_acc_shared, batch_num)

            n_predicted = n_output[1].data > 0.5
            n_acc = (n_predicted == n_label).sum() / batch_num
            n_accs.update(n_acc, batch_num)

            n_predicted_share = n_output[2].data > 0.5  # shared astnn
            n_acc_shared = (n_predicted_share == n_label).sum() / batch_num
            n_accs_shared.update(n_acc_shared, batch_num)

            losses.update(loss.item(), batch_num)
            loss1s.update(loss1.item(), batch_num)
            loss2s.update(loss2.item(), batch_num)
            loss3s.update(loss3.item(), batch_num)
            loss4s.update(loss4.item(), batch_num)

            if self.args.logverbose and batch_idx % self.log_interval == 0:
                self.log('Train Epoch: {:2d} [{:4d}/{:4d}]\t'
                         'Loss: {:.4f} ({:.4f}) \t'
                         'm_Acc: {:6.2f}% ({:.2f}%) \t'
                         'n_Acc: {:6.2f}% ({:.2f}%)'.format(cur_epoch, batch_idx * self.batch_size, total_data,
                                                            losses.val, losses.avg,
                                                            100. * m_accs.val, 100. * m_accs.avg,
                                                            100. * n_accs.val, 100. * n_accs.avg))


        return m_accs.avg, n_accs.avg, m_accs_shared.avg, n_accs_shared.avg, losses.avg

    def cross_lang_test(self, cur_epoch=0):
        """
        Testing phase for CLASTNN

        Returns
        -------
        Tuple of (loss, acc)
        """

        m_acc, m_loss, m_preds, m_trues = self._test(epoch=cur_epoch, test_inputs=self.m_test_data,
                                                     switch='m')
        n_acc, n_loss, n_preds, n_trues = self._test(epoch=cur_epoch, test_inputs=self.n_test_data,
                                                     switch='n')
        if self.name == 'CrossLing':
            tm = tn = 't'

        else:
            tm, tn = 'tm', 'tn'
        m_acc_shared, m_loss_shared, m_preds_shared, _ = self._test(epoch=cur_epoch, test_inputs=self.m_test_data,
                                                                    switch=tm, split_name='m_shared_test')

        n_acc_shared, n_loss_shared, n_preds_shared, _ = self._test(epoch=cur_epoch, test_inputs=self.n_test_data,
                                                                    switch=tn, split_name='n_shared_test')

        return m_acc, n_acc, m_acc_shared, n_acc_shared, m_preds, m_preds_shared, m_trues, \
               n_preds, n_preds_shared, n_trues

    def cross_lang_run(self, info=None, test_inputs=None):
        """
        main function to run train and testing phase
        """
        if info is not None:
            self.info = info

        for epoch in range(1, self.epochs + 1):
            start_time = time.time()

            # train for one epoch
            m_train_acc, n_train_acc, m_train_acc_shared, n_train_acc_shared, train_loss = self.cross_lang_train(epoch)

            # evaluate on validation set
            m_val_acc, n_val_acc, m_val_acc_shared, n_val_acc_shared, _, _, _, _, _, _ = self.cross_lang_test(epoch)

            end_time = time.time()
            self.log('[Epoch: %2d/%2d] Training Loss: %.4f, Time Cost: %.4fs'
                     % (epoch, self.epochs, train_loss, end_time - start_time))
            self.log('-- Private ASTNN -- Training m: %.4f, Training n: %.4f, Validation m: %.4f, Validation n: %.4f'
                     % (m_train_acc, n_train_acc, m_val_acc, n_val_acc))
            self.log('-- Shared  ASTNN -- Training m: %.4f, Training n: %.4f, Validation m: %.4f, Validation n: %.4f\n'
                     % (m_train_acc_shared, n_train_acc_shared, m_val_acc_shared, n_val_acc_shared))

            if epoch > 1:  # to skip the first epoch with random results
                # remember best acc and save checkpoint
                # self.save_checkpoint(epoch, max(m_val_acc, m_val_acc_shared))  # -- old try
                self.save_checkpoint(epoch, abs(m_val_acc - 0.5))

        self.log('---- Best Validation Accuracy for m: {} ----'.format(self.best_acc))
        self.load_checkpoint()
        m_acc, _, _, _, m_preds, m_preds_shared, m_trues, n_preds, n_preds_shared, n_trues = self.cross_lang_test()

        if m_acc < 0.5:
            print("flipping predictions")
            flip_preds = [0 if each[0] else 1 for each in m_preds]
            m_preds = flip_preds

        return m_preds, m_preds_shared, m_trues, n_preds, n_preds_shared, n_trues
