"""
author: Adnen Abdessaied
maintainer: "Adnen Abdessaied"
website: adnenabdessaied.de
version: 1.0.1
"""

# This script preprocesses clevr-dialog questions

from copy import deepcopy
from tqdm import tqdm
import numpy as np
import h5py
import json
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


parser = argparse.ArgumentParser()

parser.add_argument(
    '--input_dialogs_json',
    help='The path of the raw dialog json file.',
    required=True
    )

# '/projects/abdessaied/ns-vqa/output/clevr_vocab.json')
parser.add_argument(
    '--input_vocab_json',
    help='The path of the generated vocab.',
    required=True
)

parser.add_argument(
    '--output_vocab_json',
    help='The path to save the generated vocab.',
    required=True
)

parser.add_argument(
    '--output_h5_file',
    help='The path of the output h5 file.',
    required=True
)

parser.add_argument(
    '--mode',
    help='The preprocessing strategy.',
    choices=['stack', 'concat'],
    required=True
)

parser.add_argument(
    '--split',
    help='The split type of the data.',
    choices=['train', 'val', 'test'],
    required=True
)

parser.add_argument(
    '--percentage',
    default=1.0,
    type=int,
    help='The percentage of data to use in training.'
)

parser.add_argument(
    '--num_rounds',
    type=int,
    default=10,
    help='The total number of rounds in one dialog.'
)

parser.add_argument(
    '--val_size',
    type=int,
    help='The size of the validation set.',
    required=True
)


SPECIAL_TOKENS = {
    '<NULL>': 0,
    '<START>': 1,
    '<END>': 2,
    '<UNK>': 3,
}


def tokenize(s, delim=' ',
             add_start_token=True, add_end_token=True,
             punct_to_keep=None, punct_to_remove=None):
    """
    Tokenize a sequence, converting a string s into a list of (string) tokens by
    splitting on the specified delimiter. Optionally keep or remove certain
    punctuation marks and add start and end tokens.
    """
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, '%s%s' % (delim, p))

    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, '')

    tokens = s.split(delim)
    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')
    return tokens


def build_vocab(sequences, min_token_count=1, delim=' ',
                punct_to_keep=None, punct_to_remove=None):
    token_to_count = {}
    tokenize_kwargs = {
        'delim': delim,
        'punct_to_keep': punct_to_keep,
        'punct_to_remove': punct_to_remove,
    }
    for seq in sequences:
        seq_tokens = tokenize(seq, **tokenize_kwargs,
                              add_start_token=False, add_end_token=False)
        for token in seq_tokens:
            if token not in token_to_count:
                token_to_count[token] = 0
            token_to_count[token] += 1

    token_to_idx = {}
    for token, idx in SPECIAL_TOKENS.items():
        token_to_idx[token] = idx
    for token, count in sorted(token_to_count.items()):
        if count >= min_token_count:
            token_to_idx[token] = len(token_to_idx)

    return token_to_idx


def encode(seq_tokens, token_to_idx, allow_unk=False):
    seq_idx = []
    for token in seq_tokens:
        if token not in token_to_idx:
            if allow_unk:
                token = '<UNK>'
            else:
                raise KeyError('Token "%s" not in vocab' % token)
        seq_idx.append(token_to_idx[token])
    return seq_idx


def decode(seq_idx, idx_to_token, delim=None, stop_at_end=True):
    tokens = []
    for idx in seq_idx:
        tokens.append(idx_to_token[idx])
        if stop_at_end and tokens[-1] == '<END>':
            break
    if delim is None:
        return tokens
    else:
        return delim.join(tokens)


def concat(allDialogs, vocab, percentage, split="train", num_rounds=10):
    pbar = tqdm(allDialogs)
    pbar.set_description("[INFO] Encoding data ...")

    captions = []
    captionProgs = []
    captionImgIdx = []

    questions = []
    questionProgs = []
    questionImgIdx = []
    questionRounds = []

    histories = []
    historiesProg = []

    answers = []
    maxQ = vocab["maxQ"]
    # maxC = vocab["maxC"]
    maxP = vocab["maxP"]
    maxH = maxQ + (num_rounds-1)*(maxQ - 1)
    maxHistProg = num_rounds * maxP

    questionBins = {}
    captionBins = {}
    # k=0
    for imgDialogs in pbar:
        # k+= 1
        # if k>2:
            # break
        for dialog in imgDialogs["dialogs"]:
            if split == "train":
                if dialog["template"] not in captionBins:
                    captionBins[dialog["template"]] = {
                        "captions": [],
                        "captionProgs": []
                    }

            caption = tokenize(dialog["caption"], punct_to_keep=[
                               ';', ','], punct_to_remove=['?', '.'])

            # if len(caption) < maxQ:
            while len(caption) < maxQ:
                caption.append(vocab["text_token_to_idx"]["<NULL>"])
            caption = encode(
                caption, vocab["text_token_to_idx"], allow_unk=True)
            history = caption[:-1]  # removes <END> token

            captions.append(caption)

            progC = [dialog["template"]] + \
                list(map(lambda a: "_".join(a.split(" ")), dialog["args"]))
            progC = " ".join(progC)
            progC = tokenize(progC)
            progC = encode(progC, vocab["prog_token_to_idx"], allow_unk=True)
            while len(progC) < maxP:
                progC.append(vocab["prog_token_to_idx"]["<NULL>"])

            captionProgs.append(progC)
            imgIdx = imgDialogs["image_index"]
            captionImgIdx.append(imgIdx)

            if split == "train":
                captionBins[dialog["template"]]["captions"].append(caption)
                captionBins[dialog["template"]]["captionProgs"].append(progC)
            while len(history) < maxQ - 1:
                history.append(vocab["text_token_to_idx"]["<NULL>"])

            histoyProg = progC
            # qRounds = []
            for i, _round in enumerate(dialog["dialog"]):
                question = tokenize(_round["question"], punct_to_keep=[
                                    ';', ','], punct_to_remove=['?', '.'])
                question = encode(
                    question, vocab["text_token_to_idx"], allow_unk=True)
                questionH = question[1:-1]  # Delete <END> token

                # if len(question) < maxQ:
                # if len(question) < maxQ:
                #     print("q < {}".format(maxQ))
                # else:
                #     print("q >= {}".format(maxQ))

                while len(question) < maxQ:
                    question.append(vocab["text_token_to_idx"]["<NULL>"])
                # else:
                #     question = question[:maxQ]

                prog = [_round["template"]] + \
                    list(map(lambda a: "_".join(a.split(" ")), _round["args"]))
                prog = " ".join(prog)
                prog = tokenize(prog, punct_to_keep=[
                                ';', ','], punct_to_remove=['?', '.'])
                prog = encode(prog, vocab["prog_token_to_idx"], allow_unk=True)

                while len(prog) < maxP:
                    prog.append(vocab["prog_token_to_idx"]["<NULL>"])

                answer = tokenize("_".join(str(_round["answer"]).split(" ")), punct_to_keep=[
                                  ';', ','], punct_to_remove=['?', '.'])
                answer = encode(
                    answer, vocab["text_token_to_idx"], allow_unk=True)
                assert len(answer) == 3  # answer = <START> ans <END>
                answer = answer[1]
                historyPadded = deepcopy(history)

                while len(historyPadded) < maxH - 1:
                    historyPadded.append(vocab["text_token_to_idx"]["<NULL>"])

                historyProgPadded = deepcopy(histoyProg)
                while len(historyProgPadded) < maxHistProg:
                    historyProgPadded.append(
                        vocab["prog_token_to_idx"]["<NULL>"])

                if split == "train":
                    questionTypeIdx = _round["template"]
                    if questionTypeIdx not in questionBins:
                        questionBins[questionTypeIdx] = {
                            "questions": [],
                            "questionProgs": [],
                            "questionImgIdx": [],
                            "questionRounds": [],

                            "histories": [],
                            "historiesProg": [],
                            "answers": [],
                        }

                    questionBins[questionTypeIdx]["questions"].append(question)
                    questionBins[questionTypeIdx]["questionProgs"].append(prog)
                    questionBins[questionTypeIdx]["questionImgIdx"].append(
                        imgIdx)
                    questionBins[questionTypeIdx]["questionRounds"].append(i+1)

                    questionBins[questionTypeIdx]["histories"].append(
                        historyPadded)
                    questionBins[questionTypeIdx]["historiesProg"].append(
                        historyProgPadded)
                    questionBins[questionTypeIdx]["answers"].append(answer)
                else:
                    questions.append(question)
                    questionProgs.append(prog)
                    histories.append(historyPadded)
                    historiesProg.append(historyProgPadded)
                    answers.append(answer)
                    questionImgIdx.append(imgIdx)
                    questionRounds.append(i+1)

                while len(questionH) < maxQ-2:
                    questionH.append(vocab["text_token_to_idx"]["<NULL>"])
                qaPair = questionH + [answer]
                history.extend(qaPair)
                histoyProg.extend(prog)

    if split == "train":
        captions = []
        captionProgs = []

        questions = []
        questionProgs = []
        questionImgIdx = []
        questionRounds = []

        histories = []
        historiesProg = []
        answers = []

        for ctype in captionBins:
            numTrSamples = int(percentage * len(captionBins[ctype]["captions"]))

            captions.extend(captionBins[ctype]["captions"][:numTrSamples])
            captionProgs.extend(
                captionBins[ctype]["captionProgs"][:numTrSamples])

        for qtype in questionBins:
            numTrSamples = int(percentage *
                               len(questionBins[qtype]["questions"]))

            questions.extend(questionBins[qtype]["questions"][:numTrSamples])
            questionProgs.extend(
                questionBins[qtype]["questionProgs"][:numTrSamples])
            questionImgIdx.extend(
                questionBins[qtype]["questionImgIdx"][:numTrSamples])
            questionRounds.extend(
                questionBins[qtype]["questionRounds"][:numTrSamples])

            histories.extend(questionBins[qtype]["histories"][:numTrSamples])
            historiesProg.extend(
                questionBins[qtype]["historiesProg"][:numTrSamples])

            answers.extend(questionBins[qtype]["answers"][:numTrSamples])

    result = {
        split: {
            "captions": captions,
            "captionProgs": captionProgs,
            # "captionImgIdx": captionImgIdx,

            "questions": questions,
            "questionProgs": questionProgs,
            "questionImgIdx": questionImgIdx,
            "questionRounds": questionRounds,

            "histories": histories,
            "historiesProg": historiesProg,
            "answers": answers,
        }
    }
    return result


def stack(allDialogs, vocab, percentage, split="train", num_rounds=10):
    pbar = tqdm(allDialogs)
    pbar.set_description("[INFO] Encoding data ...")

    captions = []
    captionProgs = []
    captionImgIdx = []

    questions = []
    questionProgs = []
    questionImgIdx = []
    questionRounds = []

    histories = []
    historiesProg = []

    answers = []

    maxQ = vocab["maxQ"]
    # maxC = vocab["maxC"]
    maxP = vocab["maxP"]
    maxHistProg = num_rounds * maxP
    questionBins = {}
    captionBins = {}

    for imgDialogs in pbar:
        for dialog in imgDialogs["dialogs"]:
            if split == "train":
                if dialog["template"] not in captionBins:
                    captionBins[dialog["template"]] = {
                        "captions": [],
                        "captionProgs": []
                    }

            caption = tokenize(dialog["caption"], punct_to_keep=[
                               ';', ','], punct_to_remove=['?', '.'])
            caption = encode(
                caption, vocab["text_token_to_idx"], allow_unk=True)
            while len(caption) < maxQ:
                caption.append(vocab["text_token_to_idx"]["<NULL>"])
            captions.append(caption)

            progC = [dialog["template"]] + \
                list(map(lambda a: "_".join(a.split(" ")), dialog["args"]))
            progC = " ".join(progC)
            progC = tokenize(progC)
            progC = encode(progC, vocab["prog_token_to_idx"], allow_unk=True)
            while len(progC) < maxP:
                progC.append(vocab["prog_token_to_idx"]["<NULL>"])

            captionProgs.append(progC)
            imgIdx = imgDialogs["image_index"]
            captionImgIdx.append(imgIdx)

            if split == "train":
                captionBins[dialog["template"]]["captions"].append(caption)
                captionBins[dialog["template"]]["captionProgs"].append(progC)

            while len(caption) < maxQ + 1:
                caption.append(vocab["text_token_to_idx"]["<NULL>"])

            history = np.zeros((num_rounds, maxQ + 1))
            history[0, :] = caption
            histoyProg = progC
            # qRounds = []
            for i, _round in enumerate(dialog["dialog"]):
                question = tokenize(_round["question"], punct_to_keep=[
                                    ';', ','], punct_to_remove=['?', '.'])
                question = encode(
                    question, vocab["text_token_to_idx"], allow_unk=True)
                questionH = question[0:-1]  # Delete <END> token

                if len(question) < maxQ:
                    while len(question) < maxQ:
                        question.append(vocab["text_token_to_idx"]["<NULL>"])
                else:
                    question = question[:maxQ]

                prog = [_round["template"]] + \
                    list(map(lambda a: "_".join(a.split(" ")), _round["args"]))
                prog = " ".join(prog)
                prog = tokenize(prog, punct_to_keep=[
                                ';', ','], punct_to_remove=['?', '.'])
                prog = encode(prog, vocab["prog_token_to_idx"], allow_unk=True)

                while len(prog) < maxP:
                    prog.append(vocab["prog_token_to_idx"]["<NULL>"])

                historyProgPadded = deepcopy(histoyProg)
                while len(historyProgPadded) < maxHistProg:
                    historyProgPadded.append(
                        vocab["prog_token_to_idx"]["<NULL>"])

                answer = tokenize("_".join(str(_round["answer"]).split(" ")), punct_to_keep=[
                                  ';', ','], punct_to_remove=['?', '.'])
                answer = encode(
                    answer, vocab["text_token_to_idx"], allow_unk=True)
                assert len(answer) == 3  # answer = <START> ans <END>
                answer = answer[1]

                if split == "train":
                    questionTypeIdx = _round["template"]
                    if questionTypeIdx not in questionBins:
                        questionBins[questionTypeIdx] = {
                            "questions": [],
                            "questionProgs": [],
                            "questionImgIdx": [],
                            "questionRounds": [],

                            "histories": [],
                            "historiesProg": [],
                            "answers": [],

                        }
                    questionBins[questionTypeIdx]["questions"].append(question)
                    questionBins[questionTypeIdx]["questionProgs"].append(prog)
                    questionBins[questionTypeIdx]["questionImgIdx"].append(
                        imgIdx)
                    questionBins[questionTypeIdx]["questionRounds"].append(i+1)

                    questionBins[questionTypeIdx]["histories"].append(
                        deepcopy(history))
                    questionBins[questionTypeIdx]["historiesProg"].append(
                        historyProgPadded)
                    questionBins[questionTypeIdx]["answers"].append(answer)
                else:
                    questions.append(question)
                    questionProgs.append(prog)
                    histories.append(deepcopy(history))
                    historiesProg.append(historyProgPadded)
                    answers.append(answer)
                    questionImgIdx.append(imgIdx)
                    questionRounds.append(i+1)

                while len(questionH) < maxQ-1:
                    questionH.append(vocab["text_token_to_idx"]["<NULL>"])
                qaPair = questionH + [answer] + \
                    [vocab["text_token_to_idx"]["<END>"]]
                if i < num_rounds - 1:
                    history[i+1, :] = qaPair
                histoyProg.extend(prog)
            # questionRounds.append(qRounds)

    if split == "train":
        captions = []
        captionProgs = []

        questions = []
        questionProgs = []
        questionImgIdx = []
        questionRounds = []

        histories = []
        historiesProg = []
        answers = []

        for ctype in captionBins:
            numTrSamples = int(
                percentage * len(captionBins[ctype]["captions"]))

            captions.extend(captionBins[ctype]["captions"][:numTrSamples])
            captionProgs.extend(
                captionBins[ctype]["captionProgs"][:numTrSamples])

        for qtype in questionBins:
            numTrSamples = int(
                percentage * len(questionBins[qtype]["questions"]))

            questions.extend(questionBins[qtype]["questions"][:numTrSamples])
            questionProgs.extend(
                questionBins[qtype]["questionProgs"][:numTrSamples])
            questionImgIdx.extend(
                questionBins[qtype]["questionImgIdx"][:numTrSamples])
            questionRounds.extend(
                questionBins[qtype]["questionRounds"][:numTrSamples])

            histories.extend(questionBins[qtype]["histories"][:numTrSamples])
            historiesProg.extend(
                questionBins[qtype]["historiesProg"][:numTrSamples])

            answers.extend(questionBins[qtype]["answers"][:numTrSamples])

    result = {
        split: {
            "captions": captions,
            "captionProgs": captionProgs,

            "questions": questions,
            "questionProgs": questionProgs,
            "questionImgIdx": questionImgIdx,
            "questionRounds": questionRounds,

            "histories": histories,
            "historiesProg": historiesProg,
            "answers": answers,
        }
    }

    return result


def main(args):
    assert not((args.input_vocab_json == "")
               and (args.output_vocab_json == ""))

    print("[INFO] Loading data ...")
    with open(args.input_dialogs_json, "r") as f:
        allDialogs = json.load(f)

    # Either create the vocab or load it from disk
    if args.input_vocab_json == "":
        maxQ = 0
        maxP = 0
        text = []
        programs = []
        answers = []
        pbar = tqdm(allDialogs)
        pbar.set_description("[INFO] Building vocab ...")
        for imgDialogs in pbar:
            for dialog in imgDialogs["dialogs"]:
                text.append(dialog["caption"])
                tokenized_cap = tokenize(
                    dialog["caption"], punct_to_keep=[
                        ';', ','], punct_to_remove=['?', '.'])
                if len(tokenized_cap) > maxQ:
                    maxQ = len(tokenized_cap)

                prog = [dialog["template"]] + \
                    list(map(lambda a: "_".join(a.split(" ")), dialog["args"]))
                prog = " ".join(prog)
                programs.append(prog)
                for _round in dialog["dialog"]:
                    text.append(_round["question"])
                    tokenized_quest = tokenize(
                        _round["question"], punct_to_keep=[
                            ';', ','], punct_to_remove=['?', '.'])
                    if len(tokenized_quest) > maxQ:
                        maxQ = len(tokenized_quest)

                    prog = [_round["template"]] + \
                        list(map(lambda a: "_".join(
                            a.split(" ")), _round["args"]))
                    prog = " ".join(prog)

                    programs.append(prog)
                    answers.append("_".join(str(_round["answer"]).split(" ")))

        # print("longest question has {} tokens".format(maxQ))
        answers = list(set(answers))
        text.extend(answers)
        answer_token_to_idx = build_vocab(
            answers, punct_to_keep=[';', ','], punct_to_remove=['?', '.'])
        text_token_to_idx = build_vocab(
            text, punct_to_keep=[';', ','], punct_to_remove=['?', '.'])
        prog_token_to_idx = build_vocab(programs, punct_to_keep=[
                                        ';', ','], punct_to_remove=['?', '.'])

        idx_answer_to_token = {v: k for k, v in answer_token_to_idx.items()}
        idx_text_to_token = {v: k for k, v in text_token_to_idx.items()}
        idx_prog_to_token = {v: k for k, v in prog_token_to_idx.items()}

        vocab = {
            "text_token_to_idx": text_token_to_idx,
            "prog_token_to_idx": prog_token_to_idx,
            "answer_token_to_idx": answer_token_to_idx,
            "idx_answer_to_token": idx_answer_to_token,
            "idx_text_to_token": idx_text_to_token,
            "idx_prog_to_token": idx_prog_to_token,
            "maxQ": maxQ,
            "maxP": 6,
        }

    else:
        print("[INFO] Loading vocab ...")

        with open(args.input_vocab_json, 'r') as f:
            vocab = json.load(f)
        print("[INFO] Vocab loaded from {} ...".format(args.input_vocab_json))

    if args.output_vocab_json != "":
        if not os.path.isdir(os.path.dirname(args.output_vocab_json)):
            os.makedirs(os.path.dirname(args.output_vocab_json))
        with open(args.output_vocab_json, 'w') as f:
            json.dump(vocab, f)
        print("[INFO] Vocab saved to {} ...".format(args.output_vocab_json))

    # Encode all questions and programs
    if args.split == "train":
        if args.mode == "stack":
            result = stack(allDialogs[args.val_size:], vocab, args.percentage,
                           split=args.split, num_rounds=args.num_rounds)
        elif args.mode == "concat":
            result = concat(allDialogs[args.val_size:], vocab, args.percentage,
                            split=args.split, num_rounds=args.num_rounds)
        else:
            print("[ERROR] {} is not supported. Choose between 'concat' and 'stack'".format(
                args.mode))
            raise ValueError
    elif args.split == "val":
        if args.mode == "stack":
            result = stack(allDialogs[:args.val_size], vocab, 1.0,
                           split=args.split, num_rounds=args.num_rounds)
        elif args.mode == "concat":
            result = concat(allDialogs[:args.val_size], vocab, 1.0,
                            split=args.split, num_rounds=args.num_rounds)
        else:
            print("[ERROR] {} is not supported. Choose between 'concat' and 'stack'".format(
                args.mode))
            raise ValueError
    elif args.split == "test":
        if args.mode == "stack":
            result = stack(allDialogs, vocab, args.percentage,
                           split=args.split, num_rounds=args.num_rounds)
        elif args.mode == "concat":
            result = concat(allDialogs, vocab, args.percentage,
                            split=args.split, num_rounds=args.num_rounds)
        else:
            print("[ERROR] {} is not supported. Choose between 'concat' and 'stack'".format(
                args.mode))
            raise ValueError
    elif args.split == "finetune":
        if args.mode == "stack":
            result = stack(allDialogs, vocab, args.percentage,
                           split=args.split, num_rounds=args.num_rounds)
        elif args.mode == "concat":
            result = concat(allDialogs, vocab, args.percentage,
                            split=args.split, num_rounds=args.num_rounds)
        else:
            print("[ERROR] {} is not supported. Choose between 'concat' and 'stack'".format(
                args.mode))
            raise ValueError
    else:
        print("[ERROR] {} is not supported. Choose between 'train', 'val', and 'test'".format(
            args.mode))
        raise ValueError

    print("[INFO] Writing output ...")

    if not os.path.isdir(os.path.dirname(args.output_h5_file)):
        os.makedirs(os.path.dirname(args.output_h5_file))

    for split in result:
        if split != "train":
            args.percentage = 1.0
        with h5py.File(args.output_h5_file.format(split, args.num_rounds, args.percentage), 'w') as f:
            for dataName in result[split]:
                try:
                    data = np.asarray(result[split][dataName], dtype=np.int32)
                    f.create_dataset(dataName, data=data)
                except ValueError as e:
                    print("[INFO] Error raise by {} ...".format(dataName))
                    raise e

    print("[INFO] Done ...")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
