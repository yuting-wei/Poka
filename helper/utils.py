#!/usr/bin/env python

import codecs
import torch
import helper.logger as logger


def load_checkpoint(model_file, model, config, optimizer=None):
    """
    load models
    :param model_file: Str, file path
    :param model: Computational Graph
    :param config: helper.configure, Configure object
    :param optimizer: optimizer, torch.Adam
    :return: best_performance -> [Float, Float], config -> Configure
    """
    checkpoint_model = torch.load(model_file)
    config.train.start_epoch = checkpoint_model['epoch'] + 1
    best_performance = checkpoint_model['best_performance']
    model.load_state_dict(checkpoint_model['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_model['optimizer'])
    return best_performance, config


def save_checkpoint(state, model_file):
    """
    :param state: Dict, e.g. {'state_dict': state,
                              'optimizer': optimizer,
                              'best_performance': [Float, Float],
                              'epoch': int}
    :param model_file: Str, file path
    :return:
    """
    torch.save(state, model_file)