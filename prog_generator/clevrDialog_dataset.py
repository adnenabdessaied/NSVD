"""
author: Adnen Abdessaied
maintainer: "Adnen Abdessaied"
website: adnenabdessaied.de
version: 1.0.1
"""

import h5py
import json
import os
import numpy as np

import torch
from torch.utils.data import Dataset


def invertDict(_dict):
    return {v: k for k, v in _dict.items()}


class ClevrDialogDataset(Dataset):
    def __init__(self, dataPath, vocabPath, split, indStart=0, indEnd=-1):
        super(ClevrDialogDataset, self).__init__()
        self.data = h5py.File(dataPath, "r")
        with open(vocabPath, "r") as f:
            self.vocab = json.load(f)
        self.vocab["idx_text_to_token"] = invertDict(self.vocab["text_token_to_idx"])
        self.vocab["idx_prog_to_token"] = invertDict(self.vocab["prog_token_to_idx"])
        self.vocab["idx_prog_to_token"] = invertDict(self.vocab["prog_token_to_idx"])
        self.lenVocabText = len(self.vocab["text_token_to_idx"])
        self.lenVocabProg = len(self.vocab["prog_token_to_idx"])

        self.split = split
        self.indStart = indStart
        self.indEnd = indEnd
        self.maxSamples = indEnd - indStart
        self.maxLenProg = 6

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


class ClevrDialogCaptionDataset(ClevrDialogDataset):
    def __init__(self, dataPath, vocabPath, split, name, indStart=0, indEnd=-1):
        super(ClevrDialogCaptionDataset, self).__init__(dataPath, vocabPath, split, indStart=indStart, indEnd=indEnd)
        self.captions = torch.LongTensor(np.asarray(self.data["captions"], dtype=np.int64)[indStart: indEnd])
        self.captionsPrgs = torch.LongTensor(np.asarray(self.data["captionProgs"], dtype=np.int64)[indStart: indEnd])
        self.name = name

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        assert idx < len(self)
        caption = self.captions[idx][:16]
        captionPrg = self.captionsPrgs[idx]
        return caption, captionPrg


class ClevrDialogQuestionDataset(ClevrDialogDataset):
    def __init__(self, dataPath, vocabPath, split, name, train=True, indStart=0, indEnd=-1):
        super(ClevrDialogQuestionDataset, self).__init__(dataPath, vocabPath, split, indStart=indStart, indEnd=indEnd)
        self.questions = torch.LongTensor(np.asarray(self.data["questions"], dtype=np.int64)[indStart: indEnd])
        self.quesProgs = torch.LongTensor(np.asarray(self.data["questionProgs"], dtype=np.int64)[indStart: indEnd])
        self.questionRounds = torch.LongTensor(np.asarray(self.data["questionRounds"], dtype=np.int64)[indStart: indEnd])
        self.questionImgIdx = torch.LongTensor(np.asarray(self.data["questionImgIdx"], dtype=np.int64)[indStart: indEnd])
        self.histories = torch.LongTensor(np.asarray(self.data["histories"], dtype=np.int64)[indStart: indEnd])
        self.historiesProgs = torch.LongTensor(np.asarray(self.data["historiesProg"], dtype=np.int64)[indStart: indEnd])

        self.answers = torch.LongTensor(np.asarray(self.data["answers"], dtype=np.int64)[indStart: indEnd])
        self.name = name
        self.train = train

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        assert idx < len(self)
        question = self.questions[idx]
        questionPrg = self.quesProgs[idx]
        questionImgIdx = self.questionImgIdx[idx]
        questionRound = self.questionRounds[idx]

        history = self.histories[idx]
        historiesProg = self.historiesProgs[idx]

        answer = self.answers[idx]
        if self.train:
            return question, history, questionPrg, questionRound, answer
        else:
            return question, questionPrg, questionImgIdx, questionRound, history, historiesProg, answer
