#!/usr/bin/env python
# coding:utf-8

import torch.nn as nn
from models.kgmt import KGMT
from models.embedding_layer import EmbeddingLayer
from models.joint_learning import JL


class Poka(nn.Module):
    def __init__(self, config, vocab, model_mode='TRAIN'):
        """
        :param config: helper.configure, Configure Object
        :param vocab: data_modules.vocab, Vocab Object
        :param model_mode: Str, ('TRAIN', 'EVAL'), initialize with the pretrained word embedding if value is 'TRAIN'
        """
        super(Poka, self).__init__()
        self.config = config
        self.vocab = vocab
        self.device = config.train.device_setting.device

        self.token_map, self.label_map = vocab.v2i['token'], vocab.v2i['label']

        self.token_embedding = EmbeddingLayer(
            vocab_map=self.token_map,
            embedding_dim=config.embedding.token.dimension,
            vocab_name='token',
            config=config,
            padding_index=vocab.padding_index,
            pretrained_dir=config.embedding.token.pretrained_file,
            model_mode=model_mode,
            initial_type=config.embedding.token.init_type
        )
        print(self.token_embedding)

        self.kgmt = KGMT(config)
        print(self.kgmt)

        self.joint = JL(config=config,
                            device=self.device,
                            label_map=self.label_map)

        print(self.joint)

    def optimize_params_dict(self):
        """
        get parameters of the overall model
        :return: List[Dict{'params': Iteration[torch.Tensor],
                           'lr': Float (predefined learning rate for specified module,
                                        which is different from the others)
                          }]
        """
        params = list()
        params.append({'params': self.token_embedding.parameters()})
        params.append({'params': self.kgmt.parameters()})
        params.append({'params': self.joint.parameters()})
        return params

    def forward(self, batch):
        """
        forward pass of the overall architecture
        :param batch: DataLoader._DataLoaderIter[Dict{'token_len': List}], each batch sampled from the current epoch
        :return: 
        """

        # get distributed representation of tokens, (batch_size, max_length, embedding_dimension)
        embedding = self.token_embedding(batch['token'].to(self.config.train.device_setting.device))

        # get the length of sequences for dynamic rnn, (batch_size, 1)
        seq_len = batch['token_len']

        # get the mask matrix of theme and sentiment for attention, (batch_size, 1, max_length, max_length)
        theme_matrix = batch['mention_theme'].to(self.config.train.device_setting.device)
        senti_matrix = batch['mention_senti'].to(self.config.train.device_setting.device)

        theme_token_output, senti_token_output = self.kgmt(embedding, seq_len, theme_matrix, senti_matrix)   # ()

        theme_logits, senti_logits = self.joint(theme_token_output, senti_token_output)
        # print(logits)

        return theme_logits, senti_logits
