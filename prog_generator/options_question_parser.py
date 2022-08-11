"""
author: Adnen Abdessaied
maintainer: "Adnen Abdessaied"
website: adnenabdessaied.de
version: 1.0.1
"""
# --------------------------------------------------------
# adapted from     https://github.com/kexinyi/ns-vqa/blob/master/scene_parse/attr_net/options.py
# --------------------------------------------------------

import argparse
import os
import utils
import torch


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument(
            '--mode',
            required=True,
            type=str,
            choices=['train', 'test_with_gt', 'test_with_pred'],
            help='The mode of the experiment')

        self.parser.add_argument(
            '--run_dir',
            required=True,
            type=str,
            help='The experiment directory')

        # self.parser.add_argument('--dataset', default='clevr', type=str, help='dataset')
        self.parser.add_argument(
            '--text_log_dir',
            required=True,
            type=str,
            help='File to save the logged text')

        self.parser.add_argument(
            '--questionNetPath',
            default='',
            type=str,
            help='Path to the pretrained QuestionNet that will be used for testing.')

        self.parser.add_argument(
            '--captionNetPath',
            default='',
            type=str,
            help='Path to the pretrained CaptionNet that will be used for testing.')

        self.parser.add_argument(
            '--dialogLen',
            default=10,
            type=int,
            help='Length of the dialogs to be used for testing. We used 10, 15, and 20 in our experiments.')

        self.parser.add_argument(
            '--last_n_rounds',
            default=10,
            type=int,
            help='Number of the last rounds to consider in the history. We used 1, 2, 3, 4, and 10 in our experiments. ')

        self.parser.add_argument(
            '--encoderType',
            required=True,
            type=int,
            choices=[1, 2],
            help='Type of the encoder: 1 --> Concat, 2 --> Stack')

        self.parser.add_argument(
            '--load_checkpoint_path',
            default='None',
            type=str,
            help='Path to a QestionNet checkpoint path to resume training')

        self.parser.add_argument(
            '--gpu_ids',
            default='0',
            type=str,
            help='Id of the gpu to be used')

        self.parser.add_argument(
            '--seed',
            default=42,
            type=int,
            help='The seed used in training')

        self.parser.add_argument(
            '--dataPathTr',
            required=True,
            type=str,
            help='Path to the h5 file of the Clevr-Dialog preprocessed training data')

        self.parser.add_argument(
            '--dataPathVal',
            required=True,
            type=str,
            help='Path to the h5 file of the Clevr-Dialog preprocessed validation data')

        self.parser.add_argument(
            '--dataPathTest',
            required=True,
            type=str,
            help='Path to the h5 file of the Clevr-Dialog preprocessed test data')

        self.parser.add_argument(
            '--scenesPath',
            required=True,
            type=str,
            help='Path to the derendered clevr-dialog scenes')

        self.parser.add_argument(
            '--vocabPath',
            required=True,
            type=str,
            help='Path to the generated vocabulary')

        self.parser.add_argument(
            '--batch_size',
            default=64,
            type=int,
            help='Batch size')

        self.parser.add_argument(
            '--countFirstFailueRound',
            default=0,
            type=int,
            help='If activated, we count the first failure round')

        self.parser.add_argument(
            '--maxSamples',
            default=-1,
            type=int,
            help='Maximum number of training samples')

        self.parser.add_argument(
            '--num_workers',
            default=0,
            type=int,
            help='Number of workers for loading')

        self.parser.add_argument(
            '--num_iters',
            default=5000,
            type=int,
            help='Total number of iterations')

        self.parser.add_argument(
            '--display_every',
            default=5,
            type=int,
            help='Display training information every N iterations')

        self.parser.add_argument(
            '--validate_every',
            default=1000,
            type=int,
            help='Validate every N iterations')

        self.parser.add_argument(
            '--shuffle_data',
            default=1,
            type=int,
            help='Activate to shuffle the training data')

        self.parser.add_argument(
            '--optim',
            default='adam',
            type=str,
            help='The name of the optimizer to be used')

        self.parser.add_argument(
            '--lr',
            default=1e-3,
            type=float,
            help='Base learning rate')

        self.parser.add_argument(
            '--betas',
            default='0.9, 0.98',
            type=str,
            help='Adam optimizer\'s betas')

        self.parser.add_argument(
            '--eps',
            default='1e-9',
            type=float,
            help='Adam optimizer\'s epsilon')

        self.parser.add_argument(
            '--lr_decay_marks',
            default='50000, 55000',
            type=str,
            help='Learing rate decay marks')

        self.parser.add_argument(
            '--lr_decay_factor',
            default=0.5,
            type=float,
            help='Learning rate decay factor')

        self.parser.add_argument(
            '--weight_decay',
            default=1e-6,
            type=float,
            help='Weight decay')

        self.parser.add_argument(
            '--embedDim',
            default=300,
            type=int,
            help='Embedding dimension')

        self.parser.add_argument(
            '--hiddenDim',
            default=512,
            type=int,
            help='LSTM hidden dimension')

        self.parser.add_argument(
            '--numLayers',
            default=2,
            type=int,
            help='Number of hidden LSTM layers')

        self.parser.add_argument(
            '--dropout',
            default=0.1,
            type=float,
            help='Dropout value')

        self.parser.add_argument(
            '--multiHead',
            default=8,
            type=int,
            help='Number of attention heads')

        self.parser.add_argument(
            '--hiddenSizeHead',
            default=64,
            type=int,
            help='Dimension of each attention head')

        self.parser.add_argument(
            '--FeedForwardSize',
            default=2048,
            type=int,
            help='Dimension of the feed forward layer')

        self.parser.add_argument(
            '--FlatMLPSize',
            default=512,
            type=int,
            help='MLP flatten size')

        self.parser.add_argument(
            '--FlatGlimpses',
            default=1,
            type=int,
            help='Number of flatten glimpses')

        self.parser.add_argument(
            '--FlatOutSize',
            default=512,
            type=int,
            help='Final attention reduction dimension')

        self.parser.add_argument(
            '--layers',
            default=6,
            type=int,
            help='Number of self attention layers')

        self.parser.add_argument(
            '--bidirectional',
            default=1,
            type=int,
            help='Activate to use bidirectional LSTMs')

        self.initialized = True

    def parse(self):
        # initialize parser
        if not self.initialized:
            self.initialize()
        self.opts = self.parser.parse_args()

        # parse gpu id list
        str_gpu_ids = self.opts.gpu_ids.split(',')
        self.opts.gpu_ids = []
        for str_id in str_gpu_ids:
            if str_id.isdigit() and int(str_id) >= 0:
                self.opts.gpu_ids.append(int(str_id))
        if len(self.opts.gpu_ids) > 0 and torch.cuda.is_available():
            print('\n[INFO] Using {} CUDA device(s) ...'.format(
                len(self.opts.gpu_ids)))
        else:
            print('\n[INFO] Using cpu ...')
            self.opts.gpu_ids = []

        # parse the optimizer's betas and lr decay marks
        self.opts.betas = [float(beta) for beta in self.opts.betas.split(',')]
        lr_decay_marks = [int(m) for m in self.opts.lr_decay_marks.split(',')]
        for i in range(1, len(lr_decay_marks)):
            assert lr_decay_marks[i] > lr_decay_marks[i-1]
        self.opts.lr_decay_marks = lr_decay_marks

        # print and save options
        args = vars(self.opts)
        print('\n ' + 30*'-' + 'Opts' + 30*'-')
        for k, v in args.items():
            print('%s: %s' % (str(k), str(v)))

        if not os.path.isdir(self.opts.run_dir):
            os.makedirs(self.opts.run_dir)
        filename = 'opts.txt'
        file_path = os.path.join(self.opts.run_dir, filename)
        with open(file_path, 'wt') as fout:
            fout.write('| options\n')
            for k, v in sorted(args.items()):
                fout.write('%s: %s\n' % (str(k), str(v)))
        return self.opts
