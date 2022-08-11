"""
author: Adnen Abdessaied
maintainer: "Adnen Abdessaied"
website: adnenabdessaied.de
version: 1.0.1
"""

import os
import sys
import json, torch, pickle, copy, time
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from copy import deepcopy
from clevrDialog_dataset import ClevrDialogQuestionDataset
import pickle
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from executor.symbolic_executor import SymbolicExecutorClevr, SymbolicExecutorMinecraft
from models import SeqToSeqQ, QuestEncoder_1, QuestEncoder_2, Decoder, CaptionEncoder, SeqToSeqC
from optim import get_optim, adjust_lr
from options_caption_parser import Options as OptionsC
from options_question_parser import Options as OptionsQ


class Execution:
    def __init__(self, optsQ, optsC):
        self.opts = deepcopy(optsQ)
        if self.opts.useCuda > 0 and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("[INFO] Using GPU {} ...".format(torch.cuda.get_device_name(0)))
        else:
            print("[INFO] Using CPU ...")
            self.device = torch.device("cpu")

        self.loss_fn = torch.nn.NLLLoss().to(self.device)

        print("[INFO] Loading dataset ...")

        self.datasetTr = ClevrDialogQuestionDataset(
            self.opts.dataPathTr, self.opts.vocabPath, "train", "All tr data")

        self.datasetVal = ClevrDialogQuestionDataset(
            self.opts.dataPathVal, self.opts.vocabPath, "val", "All val data", train=False)

        self.datasetTest = ClevrDialogQuestionDataset(
            self.opts.dataPathTest, self.opts.vocabPath, "test", "All val data", train=False)

        self.QuestionNet = constructQuestionNet(
            self.opts,
            self.datasetTr.lenVocabText,
            self.datasetTr.lenVocabProg,
            self.datasetTr.maxLenProg,
            )

        if os.path.isfile(self.opts.captionNetPath):
            self.CaptionNet = constructCaptionNet(
                optsC,
                self.datasetTr.lenVocabText,
                self.datasetTr.lenVocabProg,
                self.datasetTr.maxLenProg
                )
            print('Loading CaptionNet from {}'.format(self.opts.captionNetPath))
            state_dict = torch.load(self.opts.captionNetPath)['state_dict']
            self.CaptionNet.load_state_dict(state_dict)
            self.CaptionNet.to(self.device)
            total_params_cap = sum(p.numel() for p in self.CaptionNet.parameters() if p.requires_grad)
            print("The caption encoder has {} trainable parameters".format(total_params_cap))

        self.QuestionNet.to(self.device)
        # if os.path.isfile(self.opts.load_checkpoint_path):
        #     print('Loading QuestionNet from {}'.format(optsQ.load_checkpoint_path))
        #     state_dict = torch.load(self.opts.load_checkpoint_path)['state_dict']
        #     self.QuestionNet.load_state_dict(state_dict)
        total_params_quest = sum(p.numel() for p in self.QuestionNet.parameters() if p.requires_grad)
        print("The question encoder has {} trainable parameters".format(total_params_quest))

        if "minecraft" in self.opts.scenesPath:
            self.symbolicExecutor = SymbolicExecutorMinecraft(self.opts.scenesPath)
        else:
            self.symbolicExecutor = SymbolicExecutorClevr(self.opts.scenesPath)

        tb_path = os.path.join(self.opts.run_dir, "tb_logdir")
        if not os.path.isdir(tb_path):
            os.makedirs(tb_path)

        self.ckpt_path = os.path.join(self.opts.run_dir, "ckpt_dir")
        if not os.path.isdir(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        if not os.path.isdir(self.opts.text_log_dir):
            os.makedirs(self.opts.text_log_dir)

        self.writer = SummaryWriter(tb_path)
        self.iter_val = 0

        if os.path.isfile(self.opts.dependenciesPath):
            with open(self.opts.dependenciesPath, "rb") as f:
                self.dependencies = pickle.load(f)

    def train(self):
        self.QuestionNet.train()

        # Define the multi-gpu training if needed
        if len(self.opts.gpu_ids) > 1:
            self.QuestionNet = nn.DataParallel(self.QuestionNet, device_ids=self.opts.gpu_ids)

        # Load checkpoint if resume training
        if os.path.isfile(self.opts.load_checkpoint_path):
            print("[INFO] Resume trainig from ckpt {} ...".format(
                self.opts.load_checkpoint_path
            ))

            # Load the network parameters
            ckpt = torch.load(self.opts.load_checkpoint_path)
            print("[INFO] Checkpoint successfully loaded ...")
            self.QuestionNet.load_state_dict(ckpt['state_dict'])

            # Load the optimizer paramters
            optim = get_optim(self.opts, self.QuestionNet, len(self.datasetTr))  # , ckpt['optim'], lr_base=ckpt['lr_base'])
            # optim._step = int(data_size / self.__C.BATCH_SIZE * self.__C.CKPT_EPOCH)
            optim.optimizer.load_state_dict(ckpt['optimizer'])
            _iter = 0  #  ckpt['last_iter']
            epoch = 0  # ckpt['last_epoch']

        else:
            optim = get_optim(self.opts, self.QuestionNet, len(self.datasetTr))
            _iter = 0
            epoch = 0

        trainTime = 0
        bestValAcc = float("-inf")
        bestCkp = 0
        # Training loop
        while _iter < self.opts.num_iters:

            # Learning Rate Decay
            if _iter in self.opts.lr_decay_marks:
                adjust_lr(optim, self.opts.lr_decay_factor)

            # Define multi-thread dataloader
            dataloader = Data.DataLoader(
                self.datasetTr,
                batch_size=self.opts.batch_size,
                shuffle=self.opts.shuffle_data,
                num_workers=self.opts.num_workers,
            )

            # Iteration
            time_start = 0
            time_end = 0
            for batch_iter, (quest, hist, prog, questionRound, _) in enumerate(dataloader):
                time_start = time.time()
                if _iter >= self.opts.num_iters:
                    break
                quest = quest.to(self.device)
                if self.opts.last_n_rounds < 10:
                    last_n_rounds_batch = []
                    for i, r in enumerate(questionRound.tolist()):
                        startIdx = max(r - self.opts.last_n_rounds, 0)
                        endIdx = max(r, self.opts.last_n_rounds)
                        if hist.dim() == 3:
                            assert endIdx - startIdx == self.opts.last_n_rounds
                            histBatch = hist[i, :, :]
                            last_n_rounds_batch.append(histBatch[startIdx:endIdx, :])
                        elif hist.dim() == 2:
                            startIdx *= 20
                            endIdx *= 20
                            histBatch = hist[i, :]
                            temp = histBatch[startIdx:endIdx].cpu()
                            if r > self.opts.last_n_rounds:
                                last_n_rounds_batch.append(torch.cat([torch.tensor([1]), temp, torch.tensor([2])], 0))
                            else:
                                last_n_rounds_batch.append(torch.cat([temp, torch.tensor([2, 0])], 0))
                    hist = torch.stack(last_n_rounds_batch, dim=0)
                hist = hist.to(self.device)
                prog = prog.to(self.device)
                progTarget = prog.clone()
                optim.zero_grad()

                predSoftmax, _ = self.QuestionNet(quest, hist, prog[:, :-1])
                loss = self.loss_fn(
                    # predSoftmax[:, :-1, :].contiguous().view(-1, predSoftmax.size(2)),
                    predSoftmax.contiguous().view(-1, predSoftmax.size(2)),
                    progTarget[:, 1:].contiguous().view(-1))
                loss.backward()

                if _iter % self.opts.validate_every == 0 and _iter > 0:
                    valAcc = self.val()
                    if valAcc > bestValAcc:
                        bestValAcc = valAcc
                        bestCkp = _iter
                        print("\n[INFO] Checkpointing model @ iter {} with val accuracy {}\n".format(_iter, valAcc))
                        state = {
                            'state_dict': self.QuestionNet.state_dict(),
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
                    time_end = time.time()
                    trainTime += time_end-time_start

                    print("\r[CLEVR-Dialog - %s (%d | %d)][epoch %2d][iter %4d/%4d][runtime %4f] loss: %.4f, lr: %.2e" % (
                        self.datasetTr.name,
                        batch_iter,
                        len(dataloader),
                        epoch,
                        _iter,
                        self.opts.num_iters,
                        trainTime,
                        loss.cpu().data.numpy(),
                        optim._rate,
                    ), end='          ')

                optim.step()
                _iter += 1

            epoch += 1
        print("[INFO] Avg. epoch time: {} s".format(trainTime / epoch))
        print("[INFO] Best model achieved val acc. {} @ iter {}".format(bestValAcc, bestCkp))

    def val(self):
        self.QuestionNet.eval()

        total_correct = 0
        total = 0

        if len(self.opts.gpu_ids) > 1:
            self.QuestionNet = nn.DataParallel(self.QuestionNet, device_ids=self.opts.gpu_ids)
        self.QuestionNet = self.QuestionNet.eval()
        dataloader = Data.DataLoader(
            self.datasetVal,
            batch_size=self.opts.batch_size,
            shuffle=True,
            num_workers=self.opts.num_workers,
            pin_memory=False
        )
        _iterCur = 0
        _totalCur = len(dataloader)

        for step, (question, questionPrg, questionImgIdx, questionRounds, history, historiesProg, answer) in enumerate(dataloader):
            # print("\rEvaluation: [step %4d/%4d]" % (
            print("\rEvaluation: [step %4d/%4d]" % (
                step,
                int(len(dataloader)),
            ), end='          ')

            question = question.to(self.device)

            if history.dim() == 3:
                caption = history.detach()
                caption = caption[:, 0, :]
                caption = caption[:, :16].to(self.device)
            elif history.dim() == 2:
                caption = history.detach()
                caption = caption[:, :16].to(self.device)
            if self.opts.last_n_rounds is not None:
                last_n_rounds_batch = []
                for i, r in enumerate(questionRounds.tolist()):
                    startIdx = max(r - self.opts.last_n_rounds, 0)
                    endIdx = max(r, self.opts.last_n_rounds)
                    if history.dim() == 3:
                        assert endIdx - startIdx == self.opts.last_n_rounds
                        histBatch = history[i, :, :]
                        last_n_rounds_batch.append(histBatch[startIdx:endIdx, :])
                    elif history.dim() == 2:
                        startIdx *= 20
                        endIdx *= 20
                        histBatch = history[i, :]
                        temp = histBatch[startIdx:endIdx]
                        if r > self.opts.last_n_rounds:
                            last_n_rounds_batch.append(torch.cat([torch.tensor([1]), temp, torch.tensor([2])], 0))
                        else:
                            last_n_rounds_batch.append(torch.cat([temp, torch.tensor([2, 0])], 0))
                history = torch.stack(last_n_rounds_batch, dim=0)
            history = history.to(self.device)
            questionPrg = questionPrg.to(self.device)

            questProgsToksPred = self.QuestionNet.sample(question, history)
            questProgsPred = decodeProg(questProgsToksPred, self.datasetVal.vocab["idx_prog_to_token"])
            targetProgs = decodeProg(questionPrg, self.datasetVal.vocab["idx_prog_to_token"], target=True)

            correct = [1 if pred == gt else 0 for (pred, gt) in zip(questProgsPred, targetProgs)]

            correct = sum(correct)
            total_correct += correct
            total += len(targetProgs)
            self.QuestionNet.train()

        return 100.0 * (total_correct / total)

    # Evaluation
    def eval_with_gt(self):
        # Define the multi-gpu training if needed
        all_pred_answers = []
        all_gt_answers = []
        all_question_types = []
        all_penalties = []
        all_pred_programs = []
        all_gt_programs = []

        first_failure_round = 0
        total_correct = 0
        total_acc_pen = 0
        total = 0
        total_quest_prog_correct = 0

        if len(self.opts.gpu_ids) > 1:
            self.QuestionNet = nn.DataParallel(self.QuestionNet, device_ids=self.opts.gpu_ids)
        self.QuestionNet = self.QuestionNet.eval()
        self.CaptionNet = self.CaptionNet.eval()
        if self.opts.batch_size != self.opts.dialogLen:
            print("[INFO] Changed batch size from {} to {}".format(self.opts.batch_size, self.opts.dialogLen))
            self.opts.batch_size = self.opts.dialogLen
        dataloader = Data.DataLoader(
            self.datasetTest,
            batch_size=self.opts.batch_size,
            shuffle=False,
            num_workers=self.opts.num_workers,
            pin_memory=False
        )
        _iterCur = 0
        _totalCur = len(dataloader)

        for step, (question, questionPrg, questionImgIdx, questionRounds, history, historiesProg, answer) in enumerate(dataloader):
            # print("\rEvaluation: [step %4d/%4d]" % (
            #     step + 1,
            #     int(data_size / self.opts.batch_size),
            # ), end='          ')
            # if step >= 5000:
            #     break
            batchSize = question.size(0)
            question = question.to(self.device)
            # dependecy = self.dependencies[step*batchSize:(step+1)*batchSize]

            if history.dim() == 3:
                caption = history.detach()
                caption = caption[:, 0, :]
                caption = caption[:, :16].to(self.device)
            elif history.dim() == 2:
                caption = history.detach()
                caption = caption[:, :16].to(self.device)
            if self.opts.last_n_rounds < 10:
                last_n_rounds_batch = []
                for i, r in enumerate(questionRounds.tolist()):
                    startIdx = max(r - self.opts.last_n_rounds, 0)
                    endIdx = max(r, self.opts.last_n_rounds)
                    if history.dim() == 3:
                        assert endIdx - startIdx == self.opts.last_n_rounds
                        histBatch = history[i, :, :]
                        last_n_rounds_batch.append(histBatch[startIdx:endIdx, :])
                    elif history.dim() == 2:
                        startIdx *= 20
                        endIdx *= 20
                        histBatch = history[i, :]
                        temp = histBatch[startIdx:endIdx]
                        if r > self.opts.last_n_rounds:
                            last_n_rounds_batch.append(torch.cat([torch.tensor([1]), temp, torch.tensor([2])], 0))
                        else:
                            last_n_rounds_batch.append(torch.cat([temp, torch.tensor([2, 0])], 0))
                history = torch.stack(last_n_rounds_batch, dim=0)

            history = history.to(self.device)
            questionPrg = questionPrg.to(self.device)
            historiesProg = historiesProg.tolist()
            questionRounds = questionRounds.tolist()
            answer = answer.tolist()
            answers = list(map(lambda a: self.datasetTest.vocab["idx_text_to_token"][a], answer))
            questionImgIdx = questionImgIdx.tolist()
            # if "minecraft" in self.opts.scenesPath:
            #     questionImgIdx = [idx - 1 for idx in questionImgIdx]
            questProgsToksPred = self.QuestionNet.sample(question, history)
            capProgsToksPred = self.CaptionNet.sample(caption)

            questProgsPred = decodeProg(questProgsToksPred, self.datasetTest.vocab["idx_prog_to_token"])
            capProgsPred = decodeProg(capProgsToksPred, self.datasetTest.vocab["idx_prog_to_token"])

            targetProgs = decodeProg(questionPrg, self.datasetTest.vocab["idx_prog_to_token"], target=True)
            questionTypes = [targetProg[0] for targetProg in targetProgs]
            # progHistories = getProgHistories(historiesProg[0], dataset.vocab["idx_prog_to_token"])
            progHistories = [getProgHistories(progHistToks, self.datasetTest.vocab["idx_prog_to_token"]) for progHistToks in historiesProg]
            pred_answers = []
            all_pred_programs.append([capProgsPred[0]] + questProgsPred)
            all_gt_programs.append([progHistories[0]] + (targetProgs))

            for i in range(batchSize):
                # if capProgsPred[i][0] == "extreme-center":
                #     print("bla")
                # print("idx = {}".format(questionImgIdx[i]))
                ans = self.getPrediction(
                    questProgsPred[i],
                    capProgsPred[i],
                    progHistories[i],
                    questionImgIdx[i]
                )
                # if ans == "Error":
                #     print(capProgsPred[i])
                pred_answers.append(ans)
            # print(pred_answers)
            correct = [1 if pred == ans else 0 for (pred, ans) in zip(pred_answers, answers)]
            correct_prog = [1 if pred == ans else 0 for (pred, ans) in zip(questProgsPred, targetProgs)]
            idx_false = np.argwhere(np.array(correct) == 0).squeeze(-1)
            if idx_false.shape[-1] > 0:
                first_failure_round += idx_false[0] + 1
            else:
                first_failure_round += self.opts.dialogLen + 1

            correct = sum(correct)
            correct_prog = sum(correct_prog)
            total_correct += correct
            total_quest_prog_correct += correct_prog
            total += len(answers)
            all_pred_answers.append(pred_answers)
            all_gt_answers.append(answers)
            all_question_types.append(questionTypes)
            penalty = np.zeros_like(penalty)
            all_penalties.append(penalty)
            _iterCur += 1
            if _iterCur % self.opts.display_every == 0:
                print("[Evaluation] step {0} / {1} | acc. = {2:.2f}".format(
                    _iterCur, _totalCur, 100.0 * (total_correct / total)))

        ffr = 1.0 * (first_failure_round/_totalCur)/(self.opts.dialogLen + 1)

        textOut = "\n --------------- Average First Failure Round --------------- \n"
        textOut += "{} / {}".format(ffr, self.opts.dialogLen)

        # print(total_correct, total)
        accuracy = total_correct / total
        vd_acc = total_acc_pen / total
        quest_prog_acc = total_quest_prog_correct / total
        textOut += "\n --------------- Overall acc. --------------- \n"
        textOut += "{}".format(100.0 * accuracy)
        textOut += "\n --------------- Overall VD acc. --------------- \n"
        textOut += "{}".format(100.0 * vd_acc)
        textOut += "\n --------------- Question Prog. Acc --------------- \n"
        textOut += "{}".format(100.0 * quest_prog_acc)
        textOut += get_per_round_acc(
            all_pred_answers, all_gt_answers, all_penalties)

        textOut += get_per_question_type_acc(
            all_pred_answers, all_gt_answers, all_question_types, all_penalties)

        # textOut += get_per_dependency_type_acc(
        #     all_pred_answers, all_gt_answers, all_penalties)

        textOut += "\n --------------- Done --------------- \n"
        print(textOut)
        fname = self.opts.questionNetPath.split("/")[-3] + "results_{}_{}.txt".format(self.opts.last_n_rounds, self.opts.dialogLen)
        pred_answers_fname = self.opts.questionNetPath.split("/")[-3] + "_pred_answers_{}_{}.pkl".format(self.opts.last_n_rounds, self.opts.dialogLen)
        pred_answers_fname = os.path.join("/projects/abdessaied/clevr-dialog/output/pred_answers", pred_answers_fname)
        model_name = "NSVD_stack" if "stack" in self.opts.questionNetPath else "NSVD_concat"
        experiment_name = "minecraft"
        # experiment_name += "_{}".format(self.opts.dialogLen)
        prog_output_fname = os.path.join("/projects/abdessaied/clevr-dialog/output/prog_output/{}_{}.pkl".format(model_name, experiment_name))

        fpath = os.path.join(self.opts.text_log_dir, fname)
        with open(fpath, "w") as f:
            f.writelines(textOut)
        with open(pred_answers_fname, "wb") as f:
            pickle.dump(all_pred_answers, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(prog_output_fname, "wb") as f:
            pickle.dump((all_gt_programs, all_pred_programs, all_pred_answers), f, protocol=pickle.HIGHEST_PROTOCOL)

# Evaluation
    def eval_with_pred(self):
        # Define the multi-gpu training if needed
        all_pred_answers = []
        all_gt_answers = []
        all_question_types = []
        all_penalties = []

        first_failure_round = 0
        total_correct = 0
        total_acc_pen = 0
        total = 0

        samples = {}

        if len(self.opts.gpu_ids) > 1:
            self.QuestionNet = nn.DataParallel(self.QuestionNet, device_ids=self.opts.gpu_ids)
        self.QuestionNet = self.QuestionNet.eval()
        self.CaptionNet = self.CaptionNet.eval()
        if self.opts.batch_size != self.opts.dialogLen:
            print("[INFO] Changed batch size from {} to {}".format(self.opts.batch_size, self.opts.dialogLen))
            self.opts.batch_size = self.opts.dialogLen
        dataloader = Data.DataLoader(
            self.datasetTest,
            batch_size=self.opts.batch_size,
            shuffle=False,
            num_workers=self.opts.num_workers,
            pin_memory=False
        )
        _iterCur = 0
        _totalCur = len(dataloader)
        step = 0
        for step, (question, questionPrg, questionImgIdx, questionRounds, history, historiesProg, answer) in enumerate(dataloader):
            question = question.tolist()
            questions = decode(question, self.datasetTest.vocab["idx_text_to_token"], target=True)
            questions = list(map(lambda q: " ".join(q), questions))
            targetProgs = decode(questionPrg, self.datasetTest.vocab["idx_prog_to_token"], target=True)

            questionTypes = [targetProg[0] for targetProg in targetProgs]
            targetProgs = list(map(lambda q: " ".join(q), targetProgs))

            historiesProg = historiesProg.tolist()
            progHistories = [getProgHistories(progHistToks, self.datasetTest.vocab["idx_prog_to_token"]) for progHistToks in historiesProg]

            answer = answer.tolist()
            answers = list(map(lambda a: self.datasetTest.vocab["idx_text_to_token"][a], answer))
            questionImgIdx = questionImgIdx.tolist()

            if self.opts.encoderType == 2:
                histories_eval = [history[0, 0, :].tolist()]
                caption = history.detach()
                caption = caption[0, 0, :].unsqueeze(0)
                caption = caption[:, :16].to(self.device)
            elif self.opts.encoderType == 1:
                caption = history.detach()
                histories_eval = [history[0, :20].tolist()]
                caption = caption[0, :16].unsqueeze(0).to(self.device)
            cap = decode(caption, self.datasetTest.vocab["idx_text_to_token"], target=False)
            capProgToksPred = self.CaptionNet.sample(caption)
            capProgPred = decode(capProgToksPred, self.datasetTest.vocab["idx_prog_to_token"])[0]

            pred_answers = []
            pred_quest_prog = []
            for i, (q, prog_hist, img_idx) in enumerate(zip(question, progHistories, questionImgIdx)):
                _round = i + 1
                if _round <= self.opts.last_n_rounds:
                    start = 0
                else:
                    start = _round - self.opts.last_n_rounds
                end = len(histories_eval)

                quest = torch.tensor(q).unsqueeze(0).to(self.device)
                if self.opts.encoderType == 3:
                    hist = torch.stack([torch.tensor(h) for h in histories_eval[start:end]], dim=0).unsqueeze(0).to(self.device)
                elif self.opts.encoderType == 1:
                    histories_eval_copy = deepcopy(histories_eval)
                    histories_eval_copy[-1].append(self.datasetTest.vocab["text_token_to_idx"]["<END>"])
                    hist = torch.cat([torch.tensor(h) for h in histories_eval_copy[start:end]], dim=-1).unsqueeze(0).to(self.device)

                questProgsToksPred = self.QuestionNet.sample(quest, hist)
                questProgsPred = decode(questProgsToksPred, self.datasetTest.vocab["idx_prog_to_token"])[0]
                pred_quest_prog.append(" ".join(questProgsPred))
                ans = self.getPrediction(
                    questProgsPred,
                    capProgPred,
                    prog_hist,
                    img_idx
                    )
                ans_idx = self.datasetTest.vocab["text_token_to_idx"].get(
                    ans, self.datasetTest.vocab["text_token_to_idx"]["<UNK>"])
                q[q.index(self.datasetTest.vocab["text_token_to_idx"]["<END>"])] = self.datasetTest.vocab["text_token_to_idx"]["<NULL>"]
                q[-1] = self.datasetTest.vocab["text_token_to_idx"]["<END>"]
                q.insert(-1, ans_idx)
                if self.opts.encoderType == 3:
                    histories_eval.append(copy.deepcopy(q))
                elif self.opts.encoderType == 0:
                    del q[0]
                    del q[-1]
                    histories_eval.append(copy.deepcopy(q))

                pred_answers.append(ans)

            correct = [1 if pred == ans else 0 for (pred, ans) in zip(pred_answers, answers)]
            idx_false = np.argwhere(np.array(correct) == 0).squeeze(-1)
            if idx_false.shape[-1] > 0:
                first_failure_round += idx_false[0] + 1
            else:
                first_failure_round += self.opts.dialogLen + 1

            correct = sum(correct)
            total_correct += correct
            total += len(answers)
            all_pred_answers.append(pred_answers)
            all_gt_answers.append(answers)
            all_question_types.append(questionTypes)
            _iterCur += 1
            if _iterCur % self.opts.display_every == 0:
                print("[Evaluation] step {0} / {1} | acc. = {2:.2f}".format(
                    _iterCur, _totalCur, 100.0 * (total_correct / total)
                ))
            samples["{}_{}".format(questionImgIdx[0], (step % 5) + 1)] = {
                "caption": " ".join(cap[0]),
                "cap_prog_gt": " ".join(progHistories[0][0]),
                "cap_prog_pred": " ".join(capProgPred),

                "questions": questions,
                "quest_progs_gt": targetProgs,
                "quest_progs_pred": pred_quest_prog,


                "answers": answers,
                "preds": pred_answers,
                "acc": correct,
            }


        ffr = 1.0 * self.opts.dialogLen * (first_failure_round/total)

        textOut = "\n --------------- Average First Failure Round --------------- \n"
        textOut += "{} / {}".format(ffr, self.opts.dialogLen)

        # print(total_correct, total)
        accuracy = total_correct / total
        vd_acc = total_acc_pen / total
        textOut += "\n --------------- Overall acc. --------------- \n"
        textOut += "{}".format(100.0 * accuracy)
        textOut += "\n --------------- Overall VD acc. --------------- \n"
        textOut += "{}".format(100.0 * vd_acc)

        textOut += get_per_round_acc(
            all_pred_answers, all_gt_answers, all_penalties)

        textOut += get_per_question_type_acc(
            all_pred_answers, all_gt_answers, all_question_types, all_penalties)

        textOut += "\n --------------- Done --------------- \n"
        print(textOut)
        if step >= len(dataloader):
            fname = self.opts.questionNetPath.split("/")[-3] + "_results_{}_{}_{}.txt".format(self.opts.last_n_rounds, self.opts.dialogLen, self.acc_type)
            pred_answers_fname = self.opts.questionNetPath.split("/")[-3] + "_pred_answers_{}_{}.pkl".format(self.opts.last_n_rounds, self.opts.dialogLen)
            pred_answers_fname = os.path.join("/projects/abdessaied/clevr-dialog/output/pred_answers", pred_answers_fname)

            fpath = os.path.join(self.opts.text_log_dir, fname)
            with open(fpath, "w") as f:
                f.writelines(textOut)
            with open(pred_answers_fname, "wb") as f:
                pickle.dump(all_pred_answers, f, protocol=pickle.HIGHEST_PROTOCOL)

    def getPrediction(self, questProgPred, capProgPred, historyProg, imgIndex):
        self.symbolicExecutor.reset(imgIndex)
        # if round one, execute the predicted caption program first then answer the question
        if len(historyProg) == 1:
            captionFuncLabel = capProgPred[0]
            captionFuncArgs = capProgPred[1:]

            questionFuncLabel = questProgPred[0]
            questionFuncArgs = questProgPred[1:]

            try:
                _ = self.symbolicExecutor.execute(captionFuncLabel, captionFuncArgs)
            except:
                return "Error"

            try:
                predAnswer = self.symbolicExecutor.execute(questionFuncLabel, questionFuncArgs)
            except:
                return "Error"

        # If it is not the first round, we have to execute the program history and
        # then answer the question.
        else:
            questionFuncLabel = questProgPred[0]
            questionFuncArgs = questProgPred[1:]
            for prg in historyProg:
                # prg = prg.split(" ")
                FuncLabel = prg[0]
                FuncArgs = prg[1:]
                try:
                    _ = self.symbolicExecutor.execute(FuncLabel, FuncArgs)
                except:
                    return "Error"

            try:
                predAnswer = self.symbolicExecutor.execute(questionFuncLabel, questionFuncArgs)
            except:
                return "Error"
        return str(predAnswer)

    def run(self, run_mode, epoch=None):
        self.set_seed(self.opts.seed)
        if run_mode == 'train':
            self.train()
    
        elif run_mode == 'test_with_gt':
            print('Testing with gt answers in history')
            print('Loading ckpt {}'.format(self.opts.questionNetPath))
            state_dict = torch.load(self.opts.questionNetPath)['state_dict']
            self.QuestionNet.load_state_dict(state_dict)
            self.eval_with_gt()

        elif run_mode == 'test_with_pred':
            print('Testing with predicted answers in history')
            print('Loading ckpt {}'.format(self.opts.questionNetPath))
            state_dict = torch.load(self.opts.questionNetPath)['state_dict']
            self.QuestionNet.load_state_dict(state_dict)
            self.eval_with_pred()
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


def constructQuestionNet(opts, lenVocabText, lenVocabProg, maxLenProg):
    decoder = Decoder(opts, lenVocabProg, maxLenProg)
    if opts.encoderType == 1:
        encoder = QuestEncoder_1(opts, lenVocabText)
    elif opts.encoderType == 2:
        encoder = QuestEncoder_2(opts, lenVocabText)

    net = SeqToSeqQ(encoder, decoder)
    return net


def constructCaptionNet(opts, lenVocabText, lenVocabProg, maxLenProg):
    decoder = Decoder(opts, lenVocabProg, maxLenProg)
    encoder = CaptionEncoder(opts, lenVocabText)
    net = SeqToSeqC(encoder, decoder)
    return net


def getProgHistories(progHistToks, prgIdxToToken):
    progHist = []
    temp = []
    for tok in progHistToks:
        if tok not in [0, 1, 2]:
            temp.append(prgIdxToToken[tok])
            # del progHistToks[i]
        if tok == 2:
            # del progHistToks[i]
            # progHist.append(" ".join(temp))
            progHist.append(temp)
            temp = []
    return progHist


def getHistoriesFromStack(histToks, textIdxToToken):
    histories = "\n"
    temp = []
    for i, roundToks in enumerate(histToks):
        for tok in roundToks:
            if tok not in [0, 1, 2]:
                temp.append(textIdxToToken[tok])
                # del progHistToks[i]
            if tok == 2:
                # del progHistToks[i]
                if i == 0:
                    histories += " ".join(temp) + ".\n"
                else:
                    histories += " ".join(temp[:-1]) + "? | {}.\n".format(temp[-1])
                # histories.append(temp)
                temp = []
                break
    return histories


def getHistoriesFromConcat(histToks, textIdxToToken):
    histories = []
    temp = []
    for tok in histToks:
        if tok not in [0, 1, 2]:
            temp.append(textIdxToToken[tok])
            # del progHistToks[i]
        if tok == 2:
            # del progHistToks[i]
            histories.append(" ".join(temp[:-1]) + "? | {}".format(temp[-1]))
            # histories.append(temp)
            temp = []
    return histories


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
        # progsBatch.append(" ".join(prog))
        progsBatch.append(prog)
    return progsBatch


def printPred(predSoftmax, gts, prgIdxToToken):
    assert predSoftmax.size(0) == gts.size(0)
    tokens = predSoftmax.topk(1)[1].squeeze(-1)
    tokens = tokens.tolist()
    gts = gts.tolist()
    message = "\n ------------------------ \n"
    for token, gt in zip(tokens, gts):
        message += "Prediction: "
        for tok in token:
            message += prgIdxToToken.get(tok) + " "
        message += "\n Target   : "
        for tok in gt:
            message += prgIdxToToken.get(tok) + " "
        message += "\n ------------------------ \n"
    return message


def get_per_round_acc(preds, gts, penalties):
    res = {}
    for img_preds, img_gt, img_pen in zip(preds, gts, penalties):
        img_preds = list(img_preds)
        img_gt = list(img_gt)
        img_pen = list(img_pen)
        for i, (pred, gt, pen) in enumerate(zip(img_preds, img_gt, img_pen)):
            _round = str(i + 1)
            if _round not in res:
                res[_round] = {
                    "correct": 0,
                    "all": 0
                }
            res[_round]["all"] += 1
            if pred == gt:
                res[_round]["correct"] += 0.5**pen

    textOut = "\n --------------- Per round Acc --------------- \n"
    for k in res:
        textOut += "{}: {} %\n".format(k, 100.0 * (res[k]["correct"]/res[k]["all"]))
    return textOut


def get_per_question_type_acc(preds, gts, qtypes, penalties):
    res1 = {}
    res2 = {}

    for img_preds, img_gt, img_qtypes, img_pen in zip(preds, gts, qtypes, penalties):
        # img_preds = list(img_preds)
        # img_gt = list(img_gt)
        img_pen = list(img_pen)
        for pred, gt, temp, pen in zip(img_preds, img_gt, img_qtypes, img_pen):
            if temp not in res1:
                res1[temp] = {
                    "correct": 0,
                    "all": 0
                }
            temp_cat = temp.split("-")[0]
            if temp_cat not in res2:
                res2[temp_cat] = {
                    "correct": 0,
                    "all": 0
                }
            res1[temp]["all"] += 1
            res2[temp_cat]["all"] += 1

            if pred == gt:
                res1[temp]["correct"] += 0.5**pen
                res2[temp_cat]["correct"] += 0.5**pen

    textOut = "\n --------------- Per question Type Acc --------------- \n"
    for k in res1:
        textOut += "{}: {} %\n".format(k, 100.0 * (res1[k]["correct"]/res1[k]["all"]))

    textOut += "\n --------------- Per question Category Acc --------------- \n"
    for k in res2:
        textOut += "{}: {} %\n".format(k, 100.0 * (res2[k]["correct"]/res2[k]["all"]))
    return textOut


def decode(tokens, prgIdxToToken, target=False):
    if type(tokens) != list:
        tokens = tokens.tolist()

    progsBatch = []
    for token in tokens:
        prog = []
        for tok in token:
            if tok == 2:  # <END> has index 2
                break
            prog.append(prgIdxToToken.get(tok))
        if target:
            prog = prog[1:]
        # progsBatch.append(" ".join(prog))
        progsBatch.append(prog)
    return progsBatch

if __name__ == "__main__":
    optsC = OptionsC().parse()
    optsQ = OptionsQ().parse()

    exe = Execution(optsQ, optsC)
    exe.run("test")
    print("[INFO] Done ...")
