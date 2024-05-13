#!/usr/bin/env python
# coding:utf-8

import torch
from torch import nn


class JL(nn.Module):
    def __init__(self, config, label_map, device):
        """
        Joint Learning:
        :param config: helper.configure, Configure Object
        :param label_map: helper.vocab.Vocab.v2i['label'] -> Dict{str:int}
        :param graph_model: computational graph for graph model
        :param device: torch.device, config.train.device_setting.device
        """
        super(JL, self).__init__()

        self.config = config
        self.device = device
        self.label_map = label_map

        # linear transform
        self.transformation_theme = nn.Linear(config.model.linear_transformation.text_dimension,
                                        10 * config.model.linear_transformation.node_dimension)
        # linear transform
        self.transformation_emo = nn.Linear(config.model.linear_transformation.text_dimension,
                                        12 * config.model.linear_transformation.node_dimension)

        # classifier
        self.linear_theme = nn.Linear(10 * config.embedding.label.dimension,
                                10)
        self.linear_emo = nn.Linear(12 * config.embedding.label.dimension,
                                12)

        # dropout
        self.transformation_dropout = nn.Dropout(p=config.model.linear_transformation.dropout)
        self.dropout = nn.Dropout(p=config.model.classifier.dropout)

    def forward(self, theme_feature, senti_feature):
        """
        forward pass of text feature propagation
        :param text_feature ->  torch.FloatTensor, (batch_size, K0, text_dim)
        :return: logits ->  torch.FloatTensor, (batch, N)
        """
        theme_feature = torch.cat(theme_feature, 1)
        theme_feature = theme_feature.view(theme_feature.shape[0], -1) # (batch_size, K0*dim)

        senti_feature = torch.cat(senti_feature, 1)
        senti_feature = senti_feature.view(senti_feature.shape[0], -1)

        # theme predict
        theme_feature = self.transformation_dropout(self.transformation_theme(theme_feature))
        theme_feature = theme_feature.view(theme_feature.shape[0],
                                         10,
                                         self.config.model.linear_transformation.node_dimension)
        theme_logits = self.dropout(self.linear_theme(theme_feature.view(theme_feature.shape[0], -1)))

        # emotion predict
        senti_feature = self.transformation_dropout(self.transformation_emo(senti_feature))
        senti_feature = senti_feature.view(senti_feature.shape[0],
                                         12,
                                         self.config.model.linear_transformation.node_dimension)
        senti_logits = self.dropout(self.linear_emo(senti_feature.view(senti_feature.shape[0], -1)))

        return theme_logits, senti_logits
