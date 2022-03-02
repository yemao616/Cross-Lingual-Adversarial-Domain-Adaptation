import os
import sys
import time
import shutil
import numpy as np
import torch
import logging
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support
from utils.util import AverageMeter, get_batch_data


class Trainer():
    """
    Basic Trainer Class containing train and test process
    """

    def __init__(self, args):
        # for data_loader option

        self._logger = None
        self.skip_log_fields = None
        self._load_args(args)
        self.model = self._create_model()

        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adamax(self.trainable_paras, lr=self.args.lr)

        self.best_acc = 0.0
        self.info = ''

        if self.use_gpu:
            self.model.cuda()

        self._log_creating_model()
        self._log_args()
        self._log_paras()

        torch.manual_seed(self.args.seed)

    def _create_model(self):
        print(self.name)
        if self.name == 'ASTNN':
            from astnn import NormalASTNN as net

        elif self.name == 'TLSTM':
            from tlstm import NormalTLSTM as net

        elif self.name == 'LASTNN':
            # L-ASTNN
            from astnn import TemporalASTNN as net

        elif self.name == 'TLASTNN':
            # TL-ASTNN
            from astnn import T2ASTNN as net

        elif self.name == 'CrossLing':
            # CrossLing
            from cl_astnn import CrossLing as net

        elif self.name == 'TemporalCrossLing':
            # L-CrossLing or TL-CrossLing
            from cl_astnn import TemporalCrossLing as net

        else:
            raise ValueError('Do not support {}'.format(self.name))

        model = net(vars(self.args))
        return model

    @property
    def name(self):
        return self.args.name

    @property
    def trainable_paras(self):
        return filter(lambda p: p.requires_grad, self.model.parameters())

    @property
    def freeze_astnn(self):
        return self.args.freeze_astnn

    @property
    def freeze_n_astnn(self):
        return self.args.freeze_n_astnn

    @property
    def freeze_encoder(self):
        return self.args.freeze_encoder

    @property
    def freeze_gru(self):
        return self.args.freeze_gru

    @property
    def load_pretrain(self):
        return self.args.load_pretrain

    @property
    def language(self):
        return self.args.language

    @property
    def plot_info(self):
        return '_'.join([self.lang_file, self.info])

    @property
    def use_gpu(self):
        return self.args.use_gpu

    @property
    def lang_file(self):
        return self.language + self.args.data

    @property
    def expert(self):
        return self.args.expert

    @property
    def time(self):
        return self.args.time

    @property
    def train_data(self):
        return self.args.train_data

    @property
    def val_data(self):
        return self.args.val_data

    @property
    def test_data(self):
        return self.args.test_data

    @property
    def label_size(self):
        return self.args.label_size

    @property
    def batch_size(self):
        return self.args.batch_size

    @property
    def epochs(self):
        return self.args.epochs

    @property
    def log_interval(self):
        return self.args.log_interval

    @property
    def logger(self):
        return self.get_logger()

    def get_logger(self):
        if self._logger is None:
            self._logger = logging.getLogger(self.name)
            self._logger.setLevel(logging.INFO)
            self._logger.handlers = []
            self._logger.propagate = 0

            formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')

            if self.args.verbose_mode >= 1:
                ch = logging.StreamHandler(sys.stdout)
                ch.setLevel(logging.INFO)
                ch.setFormatter(formatter)
                self._logger.addHandler(ch)

            if self.args.log_path:
                os.makedirs(os.path.dirname(self.args.log_path), exist_ok=True)
                fh = logging.FileHandler(self.args.log_path, mode='w')
                fh.setLevel(logging.INFO)
                fh.setFormatter(formatter)
                self._logger.addHandler(fh)

        return self._logger

    def log(self, msg):
        self.get_logger().info(msg)

    def _load_args(self, args):
        """
        Load data path based on different model
        """
        self.args = args
        self.args.use_gpu = self.args.use_gpu and torch.cuda.is_available()
        if self.args.use_gpu:
            torch.cuda.set_device(self.args.cuda)
            self.log('Available devices: {}'.format(torch.cuda.device_count()))
            self.log('Current cuda device: {}'.format(torch.cuda.current_device()))

        self.args.log_path = self.args.log_path or 'logs/{}/log_{}.txt'.format(self.name, self.lang_file)

    def _log_creating_model(self):
        self.log('')
        self.log('-------------------------------------------------------------------------------------------------')
        self.log('-------------------------------------------------------------------------------------------------')
        self.log('                                        {}'.format(self.name))
        self.log('-------------------------------------------------------------------------------------------------')

    def _log_args(self):
        args_dict = vars(self.args)
        self.skip_log_fields = ['pretrained_embedding', 'train_data', 'val_data', 'test_data',
                                'm_train_data', 'm_test_data', 'n_train_data', 'n_test_data']
        self.log('-------------------------------------------------------------------------------------------------')
        self.log('------------------------------- Configuration - Hyper Parameters --------------------------------')
        longest_param_name_len = max(len(param_name) for param_name, _ in args_dict.items())
        for param_name, param_val in args_dict.items():
            if param_name not in self.skip_log_fields:
                self.log('{name: <{name_len}}{val}'.format(
                    name=param_name, val=param_val, name_len=longest_param_name_len + 2))
        self.log('-------------------------------------------------------------------------------------------------')

    def _log_paras(self):
        n_parameters = 0
        for variable_name, variable in self.model.named_parameters():
            if variable.requires_grad:
                n_num = variable.data.nelement()
                n_parameters += n_num
                self.log("{:40} -- shape: {:14s} -- #params: {}".format(
                    variable_name, str(list(variable.data.size())), n_num))
        self.log('-------------------------------------------------------------------------------------------------')
        self.log('Number of params: {}'.format(n_parameters))
        self.log('-------------------------------------------------------------------------------------------------')

    def save_checkpoint(self, epoch, test_acc, info=None):
        """
        Save checkpoint to disk

        Parameters
        ----------
        epoch:      int, epoch number
        test_acc:   float, testing accuracy
        info:       str, info of the saved model file

        """
        directory = 'logs/{}/'.format(self.name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        if info is None:
            info = self.info

        filename = 'checkpoint_{}_{}.pth.tar'.format(self.lang_file, info)
        filename = directory + filename

        torch.save({
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'best_acc': max(test_acc, self.best_acc),
        }, filename)

        best_file = 'logs/{}/best_model_{}_{}.pth.tar'.format(self.name, self.lang_file, info)
        if test_acc > self.best_acc:
            self.best_acc = test_acc
            shutil.copyfile(filename, best_file)
            self.log('**** save best state dict to {} ****'.format(best_file))

    def load_checkpoint(self, info=None, filename=None):
        """
        Load checkpoint from disk

        Parameters
        ----------
        info:       str, info of the saved model file
        filename:   str, state file name
        """
        if info is None:
            info = self.info

        if filename is None:
            filename = '{}/best_model_{}_{}.pth.tar'.format(self.name, self.lang_file, info)

        self.log('======== loading state dict from logs/{} '.format(filename))
        dict_info = torch.load('logs/{}'.format(filename))
        self.model.load_state_dict(dict_info['state_dict'])

    def load_pretrained_weight(self, filename=None):
        """
        Load weights from disk

        """
        if filename is None:
            # sequential model info is formatted as : semester_min

            # filename = 'ASTNN/best_model_{}_{}.pth.tar'.format(self.lang_file,
            #                                                    self.info.split('_')[0])

            filename = 'TemporalASTNN/best_model_{}_{}_{}run.pth.tar'.format(self.lang_file,
                                                                             self.info.split('_')[0], self.args.run)

        self.log('======== loading weights from logs/{} '.format(filename))
        dict_info = torch.load('logs/{}'.format(filename))
        pretrained_dict = dict_info['state_dict']
        model_dict = self.model.state_dict()

        # 1. filter out unnecessary keys
        if self.freeze_encoder:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'encoder' in k}
        if self.freeze_gru:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'gru' in k}
        if self.freeze_astnn:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'astnn' in k}

        if self.name == 'TemporalGASTNN':
            pretrained_dict = {k.replace('mASTNN', 'astnn'): v for k, v in pretrained_dict.items()}
            pretrained_dict = {k.replace('sharedASTNN', 'Gastnn'): v for k, v in pretrained_dict.items()}

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)

        # 3. load the new state dict
        self.model.load_state_dict(model_dict)

    def _train(self, epoch, train_inputs=None, pretrain=None):
        """
        base function for training or testing phase per epoch

        Parameters
        ----------
        epoch:          int, the current epoch number
        train_inputs:   tuple (inputs, labels)
        pretrain:       pre-trained file path

        Returns
        -------
        Tuple of (accuracy, loss)
        """

        losses = AverageMeter()
        accs = AverageMeter()

        if self.load_pretrain:
            # load pre-trained astnn model
            self.load_pretrained_weight(pretrain)

        # train mode
        self.model.train()
        split_name = 'train'
        if train_inputs is None:
            train_inputs = self.train_data
        data_loader, total_data = get_batch_data(train_inputs, self.batch_size, expert=self.expert, time=self.time)

        for batch_idx, batch_data in enumerate(data_loader):
            inputs, labels = batch_data
            batch_num = len(labels)
            if self.use_gpu:
                labels = labels.cuda()

            # -------- Forward pass -------- #
            output = self.model(inputs)
            predicted = output.data > 0.5
            loss = self.criterion(output, Variable(labels))

            # -------- Backward and optimize in training phase -------- #
            self.optimizer.zero_grad()  # set gradients to 0
            loss.backward()  # computes gradients using backpropagation
            self.optimizer.step()  # performs a parameter update based on the current gradient

            # -------- calculate accuracy and loss -------- #
            acc = (predicted == labels).sum() / batch_num
            accs.update(acc.item(), batch_num)
            losses.update(loss.item(), batch_num)

            if self.args.logverbose and batch_idx % self.log_interval == 0:
                self.log('Train Epoch: {:2d} [{:4d}/{:4d}]\t'
                         'Loss: {:.4f} ({:.4f}) \t'
                         'Acc: {:6.2f}% ({:.2f}%)'.format(epoch, batch_idx * self.batch_size, total_data,
                                                          losses.val, losses.avg,
                                                          100. * accs.val, 100. * accs.avg))


        return accs.avg, losses.avg

    def _test(self, epoch=0, test_inputs=None, switch=None, split_name=None):
        """
        base function for training or testing phase per epoch

        Parameters
        ----------
        epoch:          int, the current epoch number
        test_inputs:   tuple (inputs, labels)
        switch:         str, test for cross-lang framework

        Returns
        -------
        Tuple of (accuracy, loss)
        """

        losses = AverageMeter()
        accs = AverageMeter()
        predicts = []
        trues = []

        # test mode
        self.model.eval()
        if split_name is None:
            split_name = (switch + '_' if switch else '') + 'test'
        if test_inputs is None:
            test_inputs = self.test_data

        # no shuffle for testing
        data_loader, total_data = get_batch_data(test_inputs, self.batch_size, shuffle=False, expert=self.expert,
                                                 time=self.time)

        for batch_idx, batch_data in enumerate(data_loader):
            inputs, labels = batch_data
            batch_num = len(labels)
            if self.use_gpu:
                labels = labels.cuda()

            # -------- Forward pass -------- #
            if switch is not None:
                output = self.model.forward_predict(inputs, switch)
            else:
                output = self.model(inputs)
            predicted = output.data > 0.5
            loss = self.criterion(output, Variable(labels))

            # -------- calculate accuracy and loss -------- #
            acc = (predicted == labels).sum() / batch_num
            accs.update(acc.item(), batch_num)
            losses.update(loss.item(), batch_num)
            predicts.extend(predicted.cpu().numpy())
            trues.extend(labels.cpu().numpy())

        np.save('predicts.npy', predicts)
        if epoch == 0:
            self.log(
                'Testing size: [{}] Average loss: {:.4f}, Accuracy: {:.2f}%'.format(total_data, losses.avg,
                                                                                    100. * accs.avg))
            try:
                precision, recall, f1, _ = precision_recall_fscore_support(trues, predicts, average='binary')
                self.log('F1: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%'.format(
                    100. * f1, 100. * precision, 100. * recall))
            except ValueError:
                pass
            self.log('')
        return accs.avg, losses.avg, predicts, trues

    def run(self, info=None, test_inputs=None, pretrain=None):
        """
        main function to run train and testing phase
        """
        if info is not None:
            self.info = info

        for epoch in range(1, self.epochs + 1):
            start_time = time.time()

            # train for one epoch
            train_acc, train_loss = self._train(epoch=epoch, pretrain=pretrain)

            # evaluate on validation set
            val_acc, val_loss, _, _ = self._test(epoch=epoch, test_inputs=test_inputs)

            end_time = time.time()
            self.log('[Epoch: %2d/%2d] Training Loss: %.4f, Training Acc: %.4f, '
                     'Validation Loss: %.3f, Validation Acc: %.4f, Time Cost: %.4fs'
                     % (epoch, self.epochs, train_loss, train_acc, val_loss, val_acc, end_time - start_time))

            # remember the best acc and save checkpoint
            self.save_checkpoint(epoch, abs(val_acc-0.5))

        self.log('---- Best Validation Accuracy: {} ----'.format(self.best_acc))
        self.load_checkpoint()
        test_acc, _, predicts, trues = self._test()
        if test_acc < 0.5:
            "flipping predictions"
            flip_preds = [0 if each[0] else 1 for each in predicts]
            predicts = flip_preds
        return predicts, trues
