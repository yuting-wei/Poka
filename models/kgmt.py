#!/usr/bin/env python
# coding:utf-8

import torch
from torch import nn
import torch.nn.functional as F
import math


class GRU(nn.Module):
    def __init__(self,
                 layers,
                 input_dim,
                 output_dim,
                 bias=True,
                 batch_first=False,
                 dropout=0.0,
                 bidirectional=True):
        """
        GRU module
        :param layers: int, the number of layers, config.kgmt.RNN.num_layers
        :param input_dim: int, config.embedding.token.dimension
        :param output_dim: int, config.kgmt.RNN.hidden_dimension
        :param bias: None
        :param batch_first: True
        :param dropout: p = dropout, config.kgmt.RNN.dropout
        :param bidirectional: Boolean , default True, config.kgmt.RNN.bidirectional
        """
        super(GRU, self).__init__()
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_layers = layers
        self.gru = torch.nn.GRU(input_size=input_dim,
                                hidden_size=output_dim,
                                num_layers=layers,
                                batch_first=batch_first,
                                bias=bias,
                                bidirectional=bidirectional,
                                dropout=dropout)

    def forward(self, inputs, seq_len=None, init_state=None, ori_state=False):
        """
        :param inputs: torch.FloatTensor, (batch, max_length, embedding_dim)
        :param seq_len: torch.LongTensor, (batch, max_length)
        :param init_state: None
        :param ori_state: False
        :return: padding_out -> (batch, max_length, 2 * hidden_dimension),
        """
        if seq_len is not None:
            seq_len = seq_len.int()
            sorted_seq_len, indices = torch.sort(seq_len, descending=True) 

            if self.batch_first:
                sorted_inputs = inputs[indices] # sort according to indices
            else:
                sorted_inputs = inputs[:, indices]

            packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(
                sorted_inputs,
                sorted_seq_len,
                batch_first=self.batch_first,
            ) # pack to same length

            outputs, states = self.gru(packed_inputs, init_state)
        if ori_state:
            return outputs, states
        if self.bidirectional:
            last_layer_hidden_state = states[2 * (self.num_layers - 1):]
            last_layer_hidden_state = torch.cat((last_layer_hidden_state[0], last_layer_hidden_state[1]), 1)
        else:
            last_layer_hidden_state = states[self.num_layers - 1]
            last_layer_hidden_state = last_layer_hidden_state[0]

        _, reversed_indices = torch.sort(indices, descending=False)
        last_layer_hidden_state = last_layer_hidden_state[reversed_indices]
        padding_out, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs,
                                                                batch_first=self.batch_first)
        if self.batch_first:
            padding_out = padding_out[reversed_indices]
        else:
            padding_out = padding_out[:, reversed_indices]
        return padding_out, last_layer_hidden_state

class MultiHeadedAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, hidden_size, heads_num, dropout):
        super(MultiHeadedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.per_head_size = hidden_size // heads_num

        self.linear_layers = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size) for _ in range(3)
            ])
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, key, value, query, mask, i):
        """
        Args:
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            query: [batch_size x seq_length x hidden_size]
            mask: [batch_size x 1 x seq_length x seq_length]

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, hidden_size = key.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size

        def shape(x):
            return x. \
                   contiguous(). \
                   view(batch_size, seq_length, heads_num, per_head_size). \
                   transpose(1, 2)

        def unshape(x):
            return x. \
                   transpose(1, 2). \
                   contiguous(). \
                   view(batch_size, seq_length, hidden_size)

        query, key, value = [l(x). \
                             view(batch_size, -1, heads_num, per_head_size). \
                             transpose(1, 2) \
                             for l, x in zip(self.linear_layers, (query, key, value))
                            ]

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(float(per_head_size)) 
        scores = scores * mask
        _probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(_probs)
        output = unshape(torch.matmul(probs, value))
        output = self.final_linear(output)
        
        return output

class KGMT(nn.Module):
    def __init__(self, config):
        """
        Knowledge-Guided Mask-Transformer
        :param config: helper.configure, Configure Object
        """
        super(KGMT, self).__init__()
        self.config = config
        self.layers_num = config.kgmt.att.layers_num
        self.rnn = GRU(
            layers=config.kgmt.RNN.num_layers,
            input_dim=config.embedding.token.dimension,
            output_dim=config.kgmt.RNN.hidden_dimension,
            batch_first=True,
            bidirectional=config.kgmt.RNN.bidirectional
        )
        hidden_dimension = config.kgmt.RNN.hidden_dimension
        if config.kgmt.RNN.bidirectional:
            hidden_dimension *= 2
        self.rnn_dropout = torch.nn.Dropout(p=config.kgmt.RNN.dropout)

        self.self_attn = nn.ModuleList([
            MultiHeadedAttention(
                hidden_dimension, heads_num=config.kgmt.att.heads_num, dropout=config.kgmt.att.dropout 
            )
            for _ in range(self.layers_num)
        ])

        self.kernel_sizes = config.kgmt.CNN.kernel_size
        self.convs = torch.nn.ModuleList()
        for kernel_size in self.kernel_sizes:
            self.convs.append(torch.nn.Conv1d(
                hidden_dimension,
                config.kgmt.CNN.num_kernel,
                kernel_size,
                padding=kernel_size // 2
                )
            )
        self.top_k = config.kgmt.topK_max_pooling

    def forward(self, inputs, seq_lens, theme_matrix, senti_matrix):
        """
        :param inputs: torch.FloatTensor, embedding, (batch, max_len, embedding_dim)
        :param seq_lens: torch.LongTensor, (batch, max_len)
        :param theme_matrix: torch.LongTensor, (batch, 1,  max_len, max_len)
        :param senti_matrix: torch.LongTensor, (batch, 1, max_len, max_len)
        :return: 
            hidden: [batch_size x seq_length x hidden_size]
        """
        text_output, _ = self.rnn(inputs, seq_lens)
        hidden = self.rnn_dropout(text_output)

        theme_matrix = (1.0 + theme_matrix)
        senti_matrix = (1.0 + senti_matrix)
        hidden1 = hidden
        hidden2 = hidden

        for i in range(self.layers_num):
            hidden1 = self.self_attn[i](hidden1, hidden1, hidden1, theme_matrix, i)
        theme_att_emb = hidden1
        for i in range(self.layers_num):
            hidden2 = self.self_attn[i](hidden2, hidden2, hidden2, senti_matrix, i)
        senti_att_emb = hidden2

        text_output1 = theme_att_emb.transpose(1, 2)
        topk_text_outputs1 = []
        for _, conv in enumerate(self.convs):
            convolution = F.relu(conv(text_output1))
            topk_text1 = torch.topk(convolution, self.top_k)[0].view(text_output1.size(0), -1)
            topk_text1 = topk_text1.unsqueeze(1)
            topk_text_outputs1.append(topk_text1)

        text_output2 = senti_att_emb.transpose(1, 2)
        topk_text_outputs2 = []
        for _, conv in enumerate(self.convs):
            convolution = F.relu(conv(text_output2))
            topk_text2 = torch.topk(convolution, self.top_k)[0].view(text_output2.size(0), -1)
            topk_text2 = topk_text2.unsqueeze(1)
            topk_text_outputs2.append(topk_text2)
        theme_emb = topk_text_outputs1
        senti_emb = topk_text_outputs2

        return theme_emb, senti_emb
