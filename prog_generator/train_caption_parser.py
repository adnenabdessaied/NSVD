"""
author: Adnen Abdessaied
maintainer: "Adnen Abdessaied"
website: adnenabdessaied.de
version: 1.0.1
"""

from clevrDialog_dataset import ClevrDialogCaptionDataset
from models import SeqToSeqC, CaptionEncoder, Decoder
from optim import get_optim, adjust_lr
from options_caption_parser import Options
import os, json, torch, pickle, copy, time
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from tensorboardX import SummaryWriter


class Execution:
    def __init__(self, opts):
        self.opts = opts

        self.loss_fn = torch.nn.NLLLoss().cuda()

        print("[INFO] Loading dataset ...")

        self.dataset_tr = ClevrDialogCaptionDataset(
            opts.dataPathTr, opts.vocabPath, "train", "Captions Tr")

        self.dataset_val = ClevrDialogCaptionDataset(
            opts.dataPathVal, opts.vocabPath, "val", "Captions Val")

        self.dataset_test = ClevrDialogCaptionDataset(
           opts.dataPathTest, opts.vocabPath, "test", "Captions Test")

        tb_path = os.path.join(opts.run_dir, "tb_logdir")
        if not os.path.isdir(tb_path):
            os.makedirs(tb_path)

        self.ckpt_path = os.path.join(opts.run_dir, "ckpt_dir")
        if not os.path.isdir(self.ckpt_path):
            os.makedirs(self.ckpt_path)

        self.writer = SummaryWriter(tb_path)
        self.iter_val = 0
        self.bestValAcc = float("-inf")
        self.bestValIter = -1

    def constructNet(self, lenVocabText, lenVocabProg, maxLenProg, ):
        decoder = Decoder(self.opts, lenVocabProg, maxLenProg)
        encoder = CaptionEncoder(self.opts, lenVocabText)
        net = SeqToSeqC(encoder, decoder)
        return net

    def train(self, dataset, dataset_val=None):
        # Obtain needed information
        lenVocabText = dataset.lenVocabText
        lenVocabProg = dataset.lenVocabProg
        maxLenProg = dataset.maxLenProg
        net = self.constructNet(lenVocabText, lenVocabProg, maxLenProg)

        net.cuda()
        net.train()

        # Define the multi-gpu training if needed
        if len(self.opts.gpu_ids) > 1:
            net = nn.DataParallel(net, device_ids=self.opts.gpu_ids)

        # Load checkpoint if resume training
        if self.opts.load_checkpoint_path is not None:
            print("[INFO] Resume trainig from ckpt {} ...".format(
                self.opts.load_checkpoint_path
            ))

            # Load the network parameters
            ckpt = torch.load(self.opts.load_checkpoint_path)
            print("[INFO] Checkpoint successfully loaded ...")
            net.load_state_dict(ckpt['state_dict'])

            # Load the optimizer paramters
            optim = get_optim(self.opts, net, len(dataset), lr_base=ckpt['lr_base'])
            optim.optimizer.load_state_dict(ckpt['optimizer'])

        else:
            optim = get_optim(self.opts, net, len(dataset))
        _iter = 0
        epoch = 0

        # Define dataloader
        dataloader = Data.DataLoader(
            dataset,
            batch_size=self.opts.batch_size,
            shuffle=self.opts.shuffle_data,
            num_workers=self.opts.num_workers,
        )
        _iterCur = 0
        _totalCur = len(dataloader)
        # Training loop
        while _iter < self.opts.num_iters:
            # Learning Rate Decay
            if _iter in self.opts.lr_decay_marks:
                adjust_lr(optim, self.opts.lr_decay_factor)

            time_start = time.time()
            # Iteration
            for caption, captionPrg in dataloader:
                if _iter >= self.opts.num_iters:
                    break
                caption = caption.cuda()
                captionPrg = captionPrg.cuda()
                captionPrgTarget = captionPrg.clone()
                optim.zero_grad()

                predSoftmax, _ = net(caption, captionPrg)

                loss = self.loss_fn(
                    predSoftmax[:, :-1, :].contiguous().view(-1, predSoftmax.size(2)),
                    captionPrgTarget[:, 1:].contiguous().view(-1))
                loss.backward()

                # logging
                self.writer.add_scalar(
                    'train/loss',
                    loss.cpu().data.numpy(),
                    global_step=_iter)

                self.writer.add_scalar(
                    'train/lr',
                    optim._rate,
                    global_step=_iter)
                if _iter % self.opts.display_every == 0:
                    print("\r[CLEVR-Dialog - %s (%d/%4d)][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" % (
                            dataset.name,
                            _iterCur,
                            _totalCur,
                            epoch,
                            _iter,
                            self.opts.num_iters,
                            loss.cpu().data.numpy(),
                            optim._rate,
                        ), end='          ')
                optim.step()
                _iter += 1
                _iterCur += 1

                if _iter % self.opts.validate_every == 0:
                    if dataset_val is not None:
                        valAcc = self.eval(
                            net,
                            dataset_val,
                            valid=True,
                        )
                        if valAcc > self.bestValAcc:
                            self.bestValAcc = valAcc
                            self.bestValIter = _iter

                            print("[INFO] Checkpointing model @ iter {}".format(_iter))
                            state = {
                                'state_dict': net.state_dict(),
                                'optimizer': optim.optimizer.state_dict(),
                                'lr_base': optim.lr_base,
                                'optim': optim.lr_base,
                                'last_iter': _iter,
                                'last_epoch': epoch,
                            }
                            # checkpointing
                            torch.save(
                                state,
                                os.path.join(self.ckpt_path, 'ckpt_iter' + str(_iter) + '.pkl')
                            )
                    else:
                        print("[INFO] No validation dataset available")

            time_end = time.time()
            print('Finished epoch in {}s'.format(int(time_end-time_start)))
            epoch += 1

        print("[INFO] Training done. Best model had val acc. {} @ iter {}...".format(self.bestValAcc, self.bestValIter))

    # Evaluation
    def eval(self, net, dataset, valid=False):
        net = net.eval()
        data_size = len(dataset)
        dataloader = Data.DataLoader(
            dataset,
            batch_size=self.opts.batch_size,
            shuffle=False,
            num_workers=self.opts.num_workers,
            pin_memory=False
        )
        allPredictedProgs = []
        numAllProg = 0
        falsePred = 0
        for step, (caption, captionPrg) in enumerate(dataloader):
            print("\rEvaluation: [step %4d/%4d]" % (
                step,
                int(data_size / self.opts.batch_size),
            ), end='          ')
            caption = caption.cuda()
            captionPrg = captionPrg.cuda()
            tokens = net.sample(caption)
            targetProgs = decodeProg(captionPrg, dataset.vocab["idx_prog_to_token"], target=True)
            predProgs = decodeProg(tokens, dataset.vocab["idx_prog_to_token"])
            allPredictedProgs.extend(list(map(lambda s: "( {} ( {} ) ) \n".format(s[0], ", ".join(s[1:])), predProgs)))
            numAllProg += len(targetProgs)
            for targetProg, predProg in zip(targetProgs, predProgs):
                mainMod = targetProg[0] == predProg[0]
                sameLength = len(targetProg) == len(predProg)
                sameArgs = False
                if sameLength:
                    sameArgs = True
                    for argTarget in targetProg[1:]:
                        if argTarget not in predProg[1:]:
                            sameArgs = False
                            break

                if not (mainMod and sameArgs):
                    falsePred += 1
        val_acc = (1 - (falsePred / numAllProg)) * 100.0
        print("Acc: {}".format(val_acc))
        net = net.train()
        if not valid:
            with open(self.opts.res_path, "w") as f:
                f.writelines(allPredictedProgs)
            print("[INFO] Predicted caption programs logged into {}".format(self.opts.res_path))
        return val_acc

    def run(self, run_mode):
        self.set_seed(self.opts.seed)
        if run_mode == 'train':
            self.train(self.dataset_tr, self.dataset_val)

        elif run_mode == 'test':
            lenVocabText = self.dataset_test.lenVocabText
            lenVocabProg = self.dataset_test.lenVocabProg
            maxLenProg = self.dataset_test.maxLenProg
            net = self.constructNet(lenVocabText, lenVocabProg, maxLenProg)

            print('Loading ckpt {}'.format(self.opts.load_checkpoint_path))
            state_dict = torch.load(self.opts.load_checkpoint_path)['state_dict']
            net.load_state_dict(state_dict)
            net.cuda()
            self.eval(net, self.dataset_test)

        else:
            exit(-1)

    def set_seed(self, seed):
        """Sets the seed for reproducibility.
        Args:
            seed (int): The seed used
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        print('[INFO] Seed set to {}...'.format(seed))


def decodeProg(tokens, prgIdxToToken, target=False):
    tokensBatch = tokens.tolist()
    progsBatch = []
    for tokens in tokensBatch:
        prog = []
        for tok in tokens:
            if tok == 2:  # <END> has index 2
                break
            prog.append(prgIdxToToken.get(tok))
        if target:
            prog = prog[1:]
        progsBatch.append(prog)
    return progsBatch


if __name__ == "__main__":
    opts = Options().parse()
    exe = Execution(opts)
    exe.run(opts.mode)
    print("[INFO] Done ...")
