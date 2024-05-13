#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn


class Collator(object):
    def __init__(self, config, vocab):
        """
        Collator object for the collator_fn in data_modules.data_loader
        :param config: helper.configure, Configure Object
        :param vocab: data_modules.vocab, Vocab Object
        """
        super(Collator, self).__init__()
        self.device = config.train.device_setting.device
        self.label_size = len(vocab.v2i['label'].keys())
        self.max_mention_len = config.kgmt.max_length

    def _multi_hot(self, batch_labels):
        """
        :param batch_labels: label idx list of one batch, List[List[int]], e.g.  [[1,2],[0,1,3,4]]
        :return: multi-hot value for classification -> List[List[int]], e.g. [[0,1,1,0,0],[1,1,0,1,1]
        """
        batch_size = len(batch_labels)
        max_length = max([len(sample) for sample in batch_labels])
        aligned_batch_labels = []
        for sample_label in batch_labels:
            aligned_batch_labels.append(sample_label + (max_length - len(sample_label)) * [sample_label[0]])
        aligned_batch_labels = torch.Tensor(aligned_batch_labels).long()
        batch_labels_multi_hot = torch.zeros(batch_size, self.label_size).scatter_(1, aligned_batch_labels, 1)
        return batch_labels_multi_hot

    def _mention_matrix(self, batch_mentions, batch_doc_len):
        """
        :param batch_mentions: mention idx list of one batch, List[List[int]], e.g.  [[1,2],[3,4]]
        :return: mention_mask matrix -> List[List[int]], e.g. 
            [[0,1,1,0,0],
             [1,1,1,1,1],
             [1,1,1,1,1],
             [0,1,1,0,0],
             [0,1,1,0,0]]
        """
        mask_matrix_batch = []
        max_length = max(batch_doc_len)
        for i in range(len(batch_doc_len)):
            token_num = batch_doc_len[i]

            # Calculate mask matrix
            mask_matrix = torch.zeros(token_num, token_num)
            for item in batch_mentions[i]:
                if item + 1 <= token_num:
                    mask_matrix[item, :] = 1
                    mask_matrix[:, item] = 1

            src_length = batch_doc_len[i]

            if src_length < max_length:
                pad_num = max_length - src_length
                m = nn.ConstantPad2d((0, pad_num, 0, pad_num), 0)
                mask_matrix = m(mask_matrix)
                mask_matrix = torch.Tensor(mask_matrix).float()
            mask_matrix_batch.append(mask_matrix)
        mask_matrix_batch = torch.stack(mask_matrix_batch).unsqueeze(1)
        return mask_matrix_batch

    def __call__(self, batch):
        """
        transform data for training
        :param batch: Dict{'token': List[List[int]],
                           'label': List[List[int]],
                           'mention': List[List[int]],
                            'token_len': List[int]}
        :return: batch -> Dict{'token': torch.LongTensor,
                               'label': torch.FloatTensor,
                               'token_len': torch.LongTensor,
                               'label_list': List[List[int]],
                               'mention_theme': torch.LongTensor,
                               'mention_senti': torch.LongTensor,}
        """
        batch_token = []
        batch_label = []
        batch_doc_len = []
        batch_mention_senti = []
        batch_mention_theme = []

        for sample in batch:
            batch_token.append(sample['token'])
            batch_label.append(sample['label'])
            batch_doc_len.append(sample['token_len'])
            batch_mention_senti.append(sample['mention_senti'])
            batch_mention_theme.append(sample['mention_theme'])
   

        batch_token = torch.tensor(batch_token)
        batch_multi_hot_label = self._multi_hot(batch_label)
        batch_mention_theme_matrix = self._mention_matrix(batch_mention_theme, batch_doc_len)
        batch_mention_sentiment_matrix = self._mention_matrix(batch_mention_senti, batch_doc_len)
        batch_doc_len = torch.FloatTensor(batch_doc_len)

        return {
            'token': batch_token,
            'label': batch_multi_hot_label,
            'token_len': batch_doc_len,
            'label_list': batch_label,
            'mention_theme': batch_mention_theme_matrix,
            'mention_senti': batch_mention_sentiment_matrix,
        }