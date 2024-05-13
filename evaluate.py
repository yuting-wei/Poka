# !/usr/bin/env python
# coding:utf-8

import helper.logger as logger
from models.model import Poka
import torch
import sys
from helper.configure import Configure
import os
from data_modules.data_loader import data_loaders
from data_modules.vocab import Vocab
from train_modules.trainer import Trainer
from helper.utils import load_checkpoint
import codecs


def set_optimizer(config, model):
    """
    :param config: helper.configure, Configure Object
    :param model: computational graph
    :return: torch.optim
    """
    params = model.optimize_params_dict()
    if config.train.optimizer.type == 'Adam':
        return torch.optim.Adam(lr=config.train.optimizer.learning_rate,
                                params=params)
    else:
        raise TypeError("Recommend the Adam optimizer")


def evaluate(config):
    """
    :param config: helper.configure, Configure Object
    """
    # loading corpus and generate vocabulary
    corpus_vocab = Vocab(config,
                         min_freq=5,
                         max_size=50000)

    # get data
    _, _, test_loader = data_loaders(config, corpus_vocab)

    # build up model
    poka = Poka(config, corpus_vocab, model_type=config.model.type, model_mode='TRAIN')
    poka.to(config.train.device_setting.device)
    # define optimizer
    optimize = set_optimizer(config, poka)

    model_checkpoint = config.train.checkpoint.dir
    dir_list = os.listdir(model_checkpoint)
    assert len(dir_list), "No model file in checkpoint directory!!"
    assert os.path.isfile(os.path.join(model_checkpoint, config.test.best_checkpoint)), \
        "The predefined checkpoint file does not exist."
    model_file = os.path.join(model_checkpoint, config.test.best_checkpoint)
    logger.info('Loading Previous Checkpoint...')
    logger.info('Loading from {}'.format(model_file))
    _, config = load_checkpoint(model_file=model_file,
                                model=poka,
                                config=config)
    # get epoch trainer
    trainer = Trainer(model=poka,
                      optimizer=optimize,
                      vocab=corpus_vocab,
                      config=config)
    poka.eval()
    # set origin log
    performance = trainer.eval(test_loader, -1, 'TEST')
    
    return


if __name__ == "__main__":
    configs = Configure(config_json_file=sys.argv[1])

    if configs.train.device_setting.device == 'cuda':
        os.system('CUDA_VISIBLE_DEVICES=' + str(configs.train.device_setting.visible_device_list))
    else:
        os.system("CUDA_VISIBLE_DEVICES=''")
    torch.manual_seed(2023)
    torch.cuda.manual_seed(2023)
    logger.Logger(configs)

    if not os.path.isdir(configs.train.checkpoint.dir):
        os.mkdir(configs.train.checkpoint.dir)

    evaluate(configs)
