import logging
import typing as tp

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from transformers import BertModel, AutoTokenizer, AutoConfig


use_cuda = torch.cuda.is_available()
if use_cuda:
    logging.info('Using CUDA!')
    torch_t = torch.cuda

    def from_numpy(ndarray):
        return torch.from_numpy(ndarray).pin_memory().cuda(non_blocking=True)
else:
    logging.info('Not using CUDA!')
    torch_t = torch
    from torch import from_numpy


def get_bert(bert_model, tokenizer_dir, **bert_params):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,
                                              config=AutoConfig.from_pretrained(bert_model))

    bert = BertModel.from_pretrained(bert_model, **bert_params)
    return tokenizer, bert


class WordLevelBert(nn.Module):
    """
    Runs BERT on sentences and computes an embedding for each word depending on a mode:
    start: only keep first token embedding
    end: only keep last token embedding
    avg: average embeddings of all word tokens
    """
    MODES = {'start', 'end', 'avg', 'ori'}

    def __init__(self, mode: str, model: tp.Union[str, Path], tokenizer_dir: tp.Union[str, Path],
                 batch_size: int = 128, **bert_params):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f'Unknown mode: {mode}!')

        self.mode = mode
        self.bert_tokenizer, self.bert = get_bert(model, tokenizer_dir, **bert_params)
        self.dim = self.bert.config.hidden_size
        self.max_len = self.bert.embeddings.position_embeddings.num_embeddings
        self.batch_size = batch_size

        if use_cuda:
            self.cuda()

    def forward(self, sentences, include_clssep=True):
        ann_full = None
        for i in range(0, len(sentences), self.batch_size):
            ann = self.annotate(sentences[i:i + self.batch_size],
                                include_clssep=include_clssep)
            if ann_full is None:
                ann_full = ann
            else:
                ann_full = torch.cat((ann_full, ann), dim=0)
        return ann_full

    def _encode(self, sentences: tp.List[tp.List[str]], include_clssep: bool = True):
        compute_token_mask = self.mode != 'avg'

        # Each row is the token ids for a sentence, padded with zeros.
        all_input_ids = np.zeros((len(sentences), self.max_len), dtype=int)
        # Mask with 1 for real tokens and 0 for padding.
        all_input_mask = np.zeros((len(sentences), self.max_len), dtype=int)

        if compute_token_mask:
            # Mask with 1 for one token according to the mode.
            all_token_mask = np.zeros((len(sentences), self.max_len), dtype=int)

        max_sent, word_count, token_count = 0, 0, 0
        word_token_indices = {}

        for s_num, sentence in enumerate(sentences):
            tokens, token_mask = [], []

            tokens.append('[CLS]')
            if compute_token_mask:
                token_mask.append(int(include_clssep))

            if include_clssep:
                word_token_indices[word_count] = (token_count, token_count + 1)
                word_count += 1

            token_count += 1                
            
            for word in sentence:
                word_mask = []
                word_tokens = self.bert_tokenizer.tokenize(word) if self.mode != 'ori' else [word]

                if len(word_tokens) == 0:
                    logging.error(f'Unknown word: {word} in {sentence}')
                assert len(word_tokens) > 0, f'Unknown word: {word} in {sentence}'

                if compute_token_mask:
                    for _ in range(len(word_tokens)):
                        word_mask.append(0)
                    if self.mode == 'start' or self.mode == 'ori':
                        word_mask[0] = 1
                    elif self.mode == 'end':
                        word_mask[-1] = 1

                tokens.extend(word_tokens)
                token_mask.extend(word_mask)

                word_token_indices[word_count] = (token_count, token_count + len(word_tokens))
                word_count += 1
                token_count += len(word_tokens)

            tokens.append('[SEP]')
            if compute_token_mask:
                token_mask.append(int(include_clssep))

            if include_clssep:
                word_token_indices[word_count] = (token_count, token_count + 1)
                word_count += 1

            token_count += 1

            input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)

            all_input_ids[s_num, :len(input_ids)] = input_ids
            all_input_mask[s_num, :len(input_ids)] = 1

            if compute_token_mask:
                all_token_mask[s_num, :len(token_mask)] = token_mask

            max_sent = max(max_sent, len(input_ids))

        all_input_ids = all_input_ids[:, :max_sent]
        all_input_mask = all_input_mask[:, :max_sent]
        outputs = {'all_input_ids': from_numpy(np.ascontiguousarray(all_input_ids)),
                   'all_input_mask': from_numpy(np.ascontiguousarray(all_input_mask)),
                   'word_token_indices': word_token_indices}

        if compute_token_mask:
            all_token_mask = all_token_mask[:, :max_sent]
            outputs['all_token_mask'] = from_numpy(np.ascontiguousarray(all_token_mask))

        return outputs

    def annotate(self, sentences, include_clssep=True):
        """
        Input: sentences, which is a list of sentences
            Each sentence is a list of words.
            Each word is a string.
        Output: an array with dimensions (packed_len, dim).
            packed_len is the total number of words, plus 2 for each sentence
            for [CLS] and [SEP].
        """
        if include_clssep:
            packed_len = sum([(len(s) + 2) for s in sentences])
        else:
            packed_len = sum([len(s) for s in sentences])

        enc_outputs = self._encode(sentences, include_clssep)
        # A QA model does not return the last hidden state
        # hence we set output_hidden_states=True and select the last-layer
        # state from the hidden_states array.
        output = self.bert(enc_outputs['all_input_ids'],
                           attention_mask=enc_outputs['all_input_mask'],
                           output_hidden_states=True)
        #features = output['last_hidden_state']
        features = output.hidden_states[-1]

        if 'all_token_mask' in enc_outputs:
            # keep only first or last tokens
            all_token_mask = enc_outputs['all_token_mask'].to(torch.bool).unsqueeze(-1)
            features_packed = features.masked_select(all_token_mask)
            features_packed = features_packed.reshape(-1, features.shape[-1])
        else:
            # for each word, average token embeddings
            all_input_mask = enc_outputs['all_input_mask'].to(torch.bool).unsqueeze(-1)
            word_token_indices = enc_outputs['word_token_indices']
            flattened_features = features.masked_select(all_input_mask).reshape(-1, features.shape[-1])
            features_packed = torch.zeros(packed_len, features.shape[-1])

            for word in range(packed_len):
                start_idx, end_idx = word_token_indices[word]
                feature = torch.mean(flattened_features[start_idx:end_idx], axis=0)
                features_packed[word] = feature

        if features_packed.shape[0] != packed_len:
            logging.error(f'Inconsistent len\nFeatures: {features_packed.shape[0]}, Packed len: {packed_len}')

        assert features_packed.shape[0] == packed_len, "Features: {}, \
            Packed len: {}".format(features_packed.shape[0], packed_len)

        return features_packed