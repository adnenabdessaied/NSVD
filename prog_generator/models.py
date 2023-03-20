"""
author: Adnen Abdessaied
maintainer: "Adnen Abdessaied"
website: adnenabdessaied.de
version: 1.0.1
"""

import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class MHAtt(nn.Module):
    def __init__(self, opts):
        super(MHAtt, self).__init__()
        self.opts = opts

        self.linear_v = nn.Linear(opts.hiddenDim, opts.hiddenDim)
        self.linear_k = nn.Linear(opts.hiddenDim, opts.hiddenDim)
        self.linear_q = nn.Linear(opts.hiddenDim, opts.hiddenDim)
        self.linear_merge = nn.Linear(opts.hiddenDim, opts.hiddenDim)

        self.dropout = nn.Dropout(opts.dropout)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.opts.multiHead,
            self.opts.hiddenSizeHead
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.opts.multiHead,
            self.opts.hiddenSizeHead
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.opts.multiHead,
            self.opts.hiddenSizeHead
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.opts.hiddenDim
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

class FFN(nn.Module):
    def __init__(self, opts):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=opts.hiddenDim,
            mid_size=opts.FeedForwardSize,
            out_size=opts.hiddenDim,
            dropout_r=opts.dropout,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


class SA(nn.Module):
    def __init__(self, opts):
        super(SA, self).__init__()
        self.mhatt = MHAtt(opts)
        self.ffn = FFN(opts)

        self.dropout1 = nn.Dropout(opts.dropout)
        self.norm1 = LayerNorm(opts.hiddenDim)

        self.dropout2 = nn.Dropout(opts.dropout)
        self.norm2 = LayerNorm(opts.hiddenDim)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


class AttFlat(nn.Module):
    def __init__(self, opts):
        super(AttFlat, self).__init__()
        self.opts = opts

        self.mlp = MLP(
            in_size=opts.hiddenDim,
            mid_size=opts.FlatMLPSize,
            out_size=opts.FlatGlimpses,
            dropout_r=opts.dropout,
            use_relu=True
        )
        # FLAT_GLIMPSES = 1
        self.linear_merge = nn.Linear(
            opts.hiddenDim * opts.FlatGlimpses,
            opts.FlatOutSize
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.opts.FlatGlimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted

class CaptionEncoder(nn.Module):
    def __init__(self, opts, textVocabSize):
        super(CaptionEncoder, self).__init__()
        self.embedding = nn.Embedding(textVocabSize, opts.embedDim)
        bidirectional = opts.bidirectional > 0
        self.lstmC = nn.LSTM(
            input_size=opts.embedDim,
            hidden_size=opts.hiddenDim,
            num_layers=opts.numLayers,
            batch_first=True,
            bidirectional=bidirectional
        )
        if bidirectional:
            opts.hiddenDim *= 2
            opts.hiddenSizeHead *= 2
            opts.FlatOutSize *= 2

        self.attCap = nn.ModuleList([SA(opts) for _ in range(opts.layers)])
        self.attFlatCap = AttFlat(opts)
        self.fc = nn.Linear(opts.hiddenDim, opts.hiddenDim)

    def forward(self, cap, hist=None):
        capMask = self.make_mask(cap.unsqueeze(2))
        cap = self.embedding(cap)
        cap, (_, _) = self.lstmC(cap)
        capO = cap.detach().clone()

        for attC in self.attCap:
            cap = attC(cap, capMask)
        # (batchSize, 512)
        cap = self.attFlatCap(cap, capMask)
        encOut = self.fc(cap)
        return encOut, capO

class QuestEncoder_1(nn.Module):
    """
        Concat encoder
    """
    def __init__(self, opts, textVocabSize):
        super(QuestEncoder_1, self).__init__()
        bidirectional = opts.bidirectional > 0

        self.embedding = nn.Embedding(textVocabSize, opts.embedDim)
        self.lstmQ = nn.LSTM(
            input_size=opts.embedDim,
            hidden_size=opts.hiddenDim,
            num_layers=opts.numLayers,
            bidirectional=bidirectional,
            batch_first=True
        )

        self.lstmH = nn.LSTM(
            input_size=opts.embedDim,
            hidden_size=opts.hiddenDim,
            num_layers=opts.numLayers,
            bidirectional=bidirectional,
            batch_first=True)

        if bidirectional:
            opts.hiddenDim *= 2
            opts.hiddenSizeHead *= 2
            opts.FlatOutSize *= 2
        self.attQues = nn.ModuleList([SA(opts) for _ in range(opts.layers)])
        self.attHist = nn.ModuleList([SA(opts) for _ in range(opts.layers)])

        self.attFlatQuest = AttFlat(opts)
        self.fc = nn.Linear(2 * opts.hiddenDim, opts.hiddenDim)

    def forward(self, quest, hist):
        questMask = self.make_mask(quest.unsqueeze(2))
        histMask = self.make_mask(hist.unsqueeze(2))

        # quest = F.tanh(self.embedding(quest))
        quest = self.embedding(quest)

        quest, (_, _) = self.lstmQ(quest)
        questO = quest.detach().clone()

        hist = self.embedding(hist)
        hist, (_, _) = self.lstmH(hist)

        for attQ, attH in zip(self.attQues, self.attHist):
            quest = attQ(quest, questMask)
            hist = attH(hist, histMask)
        # (batchSize, 512)
        quest = self.attFlatQuest(quest, questMask)

        # hist: (batchSize, length, 512)
        attWeights = torch.sum(torch.mul(hist, quest.unsqueeze(1)), -1)
        attWeights = torch.softmax(attWeights, -1)
        hist = torch.sum(torch.mul(hist, attWeights.unsqueeze(2)), 1)
        encOut = self.fc(torch.cat([quest, hist], -1))

        return encOut, questO

    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)


class QuestEncoder_2(nn.Module):
    """
        Stack encoder
    """
    def __init__(self, opts, textVocabSize):
        super(QuestEncoder_2, self).__init__()
        bidirectional = opts.bidirectional > 0
        self.embedding = nn.Embedding(textVocabSize, opts.embedDim)
        self.lstmQ = nn.LSTM(
            input_size=opts.embedDim,
            hidden_size=opts.hiddenDim,
            num_layers=opts.numLayers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.lstmH = nn.LSTM(
            input_size=opts.embedDim,
            hidden_size=opts.hiddenDim,
            num_layers=opts.numLayers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        if bidirectional:
            opts.hiddenDim *= 2

        self.fc = nn.Linear(2 * opts.hiddenDim, opts.hiddenDim)

    def forward(self, quest, hist):

        quest = F.tanh(self.embedding(quest))
        quest, (questH, _) = self.lstmQ(quest)

        # concatenate the last hidden states from the forward and backward pass
        # of the bidirectional lstm
        lastHiddenForward = questH[1:2, :, :].squeeze(0)
        lastHiddenBackward = questH[3:4, :, :].squeeze(0)

        # questH: (batchSize, 512)
        questH = torch.cat([lastHiddenForward, lastHiddenBackward], -1)

        questO = quest.detach().clone()

        hist = F.tanh(self.embedding(hist))
        numRounds = hist.size(1)
        histFeat = []
        for i in range(numRounds):
            round_i = hist[:, i, :, :]
            _, (round_i_h, _) = self.lstmH(round_i)

            #Same as before
            lastHiddenForward = round_i_h[1:2, :, :].squeeze(0)
            lastHiddenBackward = round_i_h[3:4, :, :].squeeze(0)
            histFeat.append(torch.cat([lastHiddenForward, lastHiddenBackward], -1))

        # hist: (batchSize, rounds, 512)
        histFeat = torch.stack(histFeat, 1)
        attWeights = torch.sum(torch.mul(histFeat, questH.unsqueeze(1)), -1)
        attWeights = torch.softmax(attWeights, -1)
        histFeat = torch.sum(torch.mul(histFeat, attWeights.unsqueeze(2)), 1)
        encOut = self.fc(torch.cat([questH, histFeat], -1))
        return encOut, questO


class Decoder(nn.Module):
    def __init__(self, opts, progVocabSize, maxLen, startID=1, endID=2):
        super(Decoder, self).__init__()
        self.numLayers = opts.numLayers
        self.bidirectional = opts.bidirectional > 0
        self.maxLen = maxLen
        self.startID = startID
        self.endID = endID

        self.embedding = nn.Embedding(progVocabSize, opts.embedDim)
        self.lstmProg = nn.LSTM(
            input_size=opts.embedDim,
            hidden_size=2*opts.hiddenDim if self.bidirectional else opts.hiddenDim,
            num_layers=opts.numLayers,
            batch_first=True,
            # bidirectional=self.bidirectional,
        )
        hiddenDim = opts.hiddenDim
        if self.bidirectional:
            hiddenDim *= 2

        self.fcAtt = nn.Linear(2*hiddenDim, hiddenDim)
        self.fcOut = nn.Linear(hiddenDim, progVocabSize)

    def initPrgHidden(self, encOut):
        hidden = [encOut for _ in range(self.numLayers)]
        hidden = torch.stack(hidden, 0).contiguous()
        return hidden, hidden

    def forwardStep(self, prog, progH, questO):
        batchSize = prog.size(0)
        inputDim = questO.size(1)
        prog = self.embedding(prog)
        outProg, progH = self.lstmProg(prog, progH)

        att = torch.bmm(outProg, questO.transpose(1, 2))
        att = F.softmax(att.view(-1, inputDim), 1).view(batchSize, -1, inputDim)
        context = torch.bmm(att, questO)
        # (batchSize, progLength, hiddenDim)
        out = F.tanh(self.fcAtt(torch.cat([outProg, context], dim=-1)))

        # (batchSize, progLength, progVocabSize)
        out = self.fcOut(out)
        predSoftmax = F.log_softmax(out, 2)
        return predSoftmax, progH

    def forward(self, prog, encOut, questO):
        progH = self.initPrgHidden(encOut)
        predSoftmax, progH = self.forwardStep(prog, progH, questO)

        return predSoftmax, progH

    def sample(self, encOut, questO):
        batchSize = encOut.size(0)
        cudaFlag = encOut.is_cuda
        progH = self.initPrgHidden(encOut)
        # prog = progCopy[:, 0:3]
        prog = torch.LongTensor(batchSize, 1).fill_(self.startID)
        # prog = torch.cat((progStart, progEnd), -1)
        if cudaFlag:
            prog = prog.cuda()
        outputLogProbs = []
        outputTokens = []

        def decode(i, output):
            tokens = output.topk(1, dim=-1)[1].view(batchSize, -1)
            return tokens

        for i in range(self.maxLen):
            predSoftmax, progH = self.forwardStep(prog, progH, questO)
            prog = decode(i, predSoftmax)
    
        return outputTokens, outputLogProbs


class SeqToSeqC(nn.Module):
    def __init__(self, encoder, decoder):
        super(SeqToSeqC, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, cap, imgFeat, prog):
        encOut, capO = self.encoder(cap, imgFeat)
        predSoftmax, progHC = self.decoder(prog, encOut, capO)
        return predSoftmax, progHC
   
    def sample(self, cap):
        with torch.no_grad():
            encOut, capO = self.encoder(cap)
        outputTokens, outputLogProbs = self.decoder.sample(encOut, capO)
        outputTokens = torch.stack(outputTokens, 0).transpose(0, 1)
        return outputTokens


class SeqToSeqQ(nn.Module):
    def __init__(self, encoder, decoder):
        super(SeqToSeqQ, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, quest, hist, prog):
        encOut, questO = self.encoder(quest, hist)
        predSoftmax, progHC = self.decoder(prog, encOut, questO)
        return predSoftmax, progHC

    def sample(self, quest, hist):
        with torch.no_grad():
            encOut, questO = self.encoder(quest, hist)
            outputTokens, outputLogProbs = self.decoder.sample(encOut, questO)
        outputTokens = torch.stack(outputTokens, 0).transpose(0, 1)
        return outputTokens
