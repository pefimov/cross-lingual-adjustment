import torch
import torch.nn.functional as F
from functools import reduce
import numpy as np
import random
from word_level_bert import WordLevelBert

MIN_WGHT = 1e-10

def keep_1to1(alignments):
    if len(alignments) == 0:
        return alignments

    counts1 = np.zeros(np.max(alignments[:, 0]) + 1)
    counts2 = np.zeros(np.max(alignments[:, 1]) + 1)

    for a in alignments:
        counts1[a[0]] += 1
        counts2[a[1]] += 1

    alignments2 = []
    for a in alignments:
        if counts1[a[0]] == 1 and counts2[a[1]] == 1:
            alignments2.append(a)
    return np.array(alignments2)


def load_align_corpus(sent_path, align_path, max_len=64, max_sent=np.inf, sample_random=True):
    sentences_1 = []
    sentences_2 = []
    bad_idx = []
    with open(sent_path) as sent_file:
        """Lines should be of the form
        doch jetzt ist der Held gefallen . ||| but now the hero has fallen .

        Result: 
        [
        ['doch', 'jetzt', ...],
        ...
        ]

        [
        ['but', 'now', ...],
        ...
        ]

        If sentences are already in sub-tokenized form, then max_len should be
        512. Otherwise, sentence length might increase after bert tokenization.
        (Bert has a max length of 512.)
        """
        for i, line in enumerate(sent_file):
#             if i >= max_sent:
#                 break

            sent_1 = line[:line.index("|||")].split()
            sent_2 = line[line.index("|||"):].split()[1:]

            if len(sent_1) > max_len or len(sent_2) > max_len:
                bad_idx.append(i)
            else:
                sentences_1.append(sent_1)
                sentences_2.append(sent_2)

    if align_path is None:
        return sentences_1, sentences_2, None

    alignments = []
    with open(align_path) as align_file:
        """Lines should be of the form
        0-0 1-1 2-4 3-2 4-3 5-5 6-6

        Only keeps 1-to-1 alignments.

        Result:
        [
        [[0,0], [1,1], ...],
        ...
        ]
        """
        # need to only keep 1-1 alignments
        for i, line in enumerate(align_file):
#             if i >= max_sent:
#                 break

            if i not in bad_idx:
                alignment = [pair.split('-') for pair in line.split()]
                alignment = np.array(alignment).astype(int)
                alignment = keep_1to1(alignment)

                alignments.append(alignment)
    
    if len(alignments) < max_sent:
        max_sent = len(alignments)
    
    # if sentences had only non 1-1 alignments, corresponding would be empty
    # filter such cases to return only sentences with non-empty final alignments
    filtered_sentences_1, filtered_sentences_2, filtered_alignments = [], [], []
    if sample_random:
        triplets = random.sample(list(zip(sentences_1, sentences_2, alignments)), max_sent)
    else:
        triplets = list(zip(sentences_1, sentences_2, alignments))[:max_sent]
    for sent_1, sent_2, alignment in triplets:
        if len(alignment) == 0:
            continue
        filtered_sentences_1.append(sent_1)
        filtered_sentences_2.append(sent_2)
        filtered_alignments.append(alignment)

    return filtered_sentences_1, filtered_sentences_2, filtered_alignments


def filter_data_alignments(data, lang=None):
    filtered_data = []

    for sents_1, sents_2, alignments in data:
        filtered_sents_1, filtered_sents_2, filtered_alignments = [], [], []

        for sent_1, sent_2, alignment in zip(sents_1, sents_2, alignments):
            if alignment.size == 0:
                continue

            filtered_alignment = []

            for idx1, idx2 in alignment:
                token1 = sent_1[idx1]
                token2 = sent_2[idx2]

                if token1 != token2:
                    filtered_alignment.append([idx1, idx2])

            # if lang == 'vi':  # temporary solution for vi
            #     sent_1 = [word.replace('_', ' ') if '_' in word and word.replace('_', '') != '' else word for word in sent_1]

            if filtered_alignment:
                filtered_sents_1.append(sent_1)
                filtered_sents_2.append(sent_2)
                filtered_alignments.append(np.array(filtered_alignment))

        filtered_data.append((filtered_sents_1, filtered_sents_2, filtered_alignments))
    return filtered_data


def partial_sums(arr):
    for i in range(1, len(arr)):
        arr[i] += arr[i - 1]
    arr.insert(0, 0)
    return arr[:-1]


def pick_aligned(sent_1, sent_2, align, cls_sep=True):
    """
    sent_1, sent_2 - lists of sentences. each sentence is a list of words.
    align - lists of alignments. each alignment is a list of pairs (i,j).
    """
    idx_1 = partial_sums([len(s) + 2 for s in sent_1])
    idx_2 = partial_sums([len(s) + 2 for s in sent_2])
    align = [a + [i_1, i_2] for a, i_1, i_2 in zip(align, idx_1, idx_2)]
    align = reduce(lambda x, y: np.vstack((x, y)), align)
    align = align + 1  # to account for extra [CLS] at beginning

    if cls_sep:
        # also add cls and sep as alignments
        cls_idx = np.array(list(zip(idx_1, idx_2)))
        sep_idx = (cls_idx - 1)[1:]
        sep_idx_last = np.array([(sum([len(s) + 2 for s in sent_1]) - 1,
                                  sum([len(s) + 2 for s in sent_2]) - 1)])

        align = np.vstack((align, cls_idx, sep_idx, sep_idx_last))

    # returns idx_1, idx_2
    # pick out aligned tokens using ann_1[idx_1], ann_2[idx_2]
    return align[:, 0], align[:, 1]


def t(vecs):
    if len(vecs.shape) == 1:
        return vecs
    assert len(vecs.shape) == 2, "Expected 2D matrix, \
        got {} dimensions".format(len(vecs.shape))
    if isinstance(vecs, torch.Tensor):
        return torch.t(vecs)
    elif isinstance(vecs, np.ndarray):
        return np.transpose(vecs)
    else:
        assert False, "Type of vecs is {}, but it should be \
        torch.Tensor or np.ndarray".format(type(vecs))



class AlignSample:
    def __init__(self, indx_sample, sent_sample_1, sent_sample_2, idx_1, idx_2):
        self.indx_sample = indx_sample
        self.sent_sample_1, self.sent_sample_2 = sent_sample_1, sent_sample_2
        self.idx_1, self.idx_2 = idx_1, idx_2

class RandomAlignedSentSampler:
    def __init__(self,
                 sent_path,
                 align_path,
                 max_sent_qty,
                 lang):
        align_corpus = load_align_corpus(sent_path=sent_path,
                                         align_path=align_path,
                                         max_sent=max_sent_qty)

        filtered_data = filter_data_alignments([align_corpus], lang=lang)
        self.sent_1, self.sent_2, self.align = filtered_data[0]
        self.qty = len(self.sent_1)
        assert len(self.sent_2) == self.qty

    def sample_batch(self, batch_size, include_clssep):
        indx_sample = np.random.randint(low=0, high=self.qty, size=batch_size)
        sent_sample_1 = [self.sent_1[indx] for indx in indx_sample]
        sent_sample_2 = [self.sent_2[indx] for indx in indx_sample]
        align_sample = [self.align[indx] for indx in indx_sample]

        idx_1, idx_2 = pick_aligned(sent_sample_1, sent_sample_2, align_sample, include_clssep)

        return AlignSample(indx_sample=indx_sample,
                           sent_sample_1=sent_sample_1,
                           sent_sample_2=sent_sample_2,
                           idx_1=idx_1,
                           idx_2=idx_2)


def comp_align_loss(align_batch,
                    align_wght : float, base_diff_wght: float,
                    model_align: WordLevelBert,
                    model_align_base: WordLevelBert = None):

    if align_wght > MIN_WGHT or base_diff_wght > MIN_WGHT:
        ann_2 = model_align(align_batch.sent_sample_2)

    if align_wght > MIN_WGHT:
        ann_1 = model_align(align_batch.sent_sample_1)
        loss = align_wght * F.mse_loss(ann_1[align_batch.idx_1], ann_2[align_batch.idx_2])
    else:
        loss = 0

    if base_diff_wght > MIN_WGHT:
        with torch.no_grad():
            ann_2_base = model_align_base(align_batch.sent_sample_2)
        loss += base_diff_wght * F.mse_loss(ann_2, ann_2_base)

    return loss
