#!/usr/bin/env python
# coding:utf-8

import helper.logger as logger
from models.model import Poka
import torch
import torch.nn as nn
import sys
from helper.configure import Configure
import os
from data_modules.data_loader import data_loaders
from data_modules.vocab import Vocab
from train_modules.trainer import Trainer
from helper.utils import load_checkpoint, save_checkpoint
import time
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


def train(config):
    """
    :param config: helper.configure, Configure Object
    """
    # loading corpus and generate vocabulary
    corpus_vocab = Vocab(config,
                         min_freq=5,
                         max_size=50000)

    # get data
    train_loader, dev_loader, test_loader = data_loaders(config, corpus_vocab)

    # build up model
    poka = Poka(config, corpus_vocab, model_mode='TRAIN')
    poka.to(config.train.device_setting.device)

    # define optimizer
    optimize = set_optimizer(config, poka)

    # get epoch trainer
    trainer = Trainer(model=poka,
                      optimizer=optimize,
                      vocab=corpus_vocab,
                      config=config)

    # set origin log
    best_epoch = [-1, -1]
    best_performance = [0.0, 0.0]
    model_checkpoint = config.train.checkpoint.dir
    model_name = config.model.name
    wait = 0
    if not os.path.isdir(model_checkpoint):
        os.mkdir(model_checkpoint)
    else:
        # loading previous checkpoint
        dir_list = os.listdir(model_checkpoint)
        dir_list.sort(key=lambda fn: os.path.getatime(os.path.join(model_checkpoint, fn)))
        latest_model_file = ''
        for model_file in dir_list[::-1]:
            if model_file.startswith('best'):
                continue
            else:
                latest_model_file = model_file
                break
        if os.path.isfile(os.path.join(model_checkpoint, latest_model_file)):
            logger.info('Loading Previous Checkpoint...')
            logger.info('Loading from {}'.format(os.path.join(model_checkpoint, latest_model_file)))
            best_performance, config = load_checkpoint(model_file=os.path.join(model_checkpoint, latest_model_file),
                                                       model=poka,
                                                       config=config,
                                                       optimizer=optimize)
            logger.info('Previous Best Performance---- Micro-F1: {}%, Macro-F1: {}%'.format(
                best_performance[0], best_performance[1]))

    # train
    for epoch in range(config.train.start_epoch, config.train.end_epoch):
        start_time = time.time()
        trainer.train(train_loader,
                      epoch)
        trainer.eval(train_loader, epoch, 'TRAIN')
        performance = trainer.eval(dev_loader, epoch, 'DEV')
        with codecs.open(os.path.join(model_checkpoint, 'candidate.txt'),'w+','utf-8') as f:
            for i in range(len(performance['gold_label'])):
                f.write(" ".join(performance['gold_label'][i])+'\n')
        # saving best model and check model
        if not (performance['micro_f1'] >= best_performance[0] or performance['macro_f1'] >= best_performance[1]):
            wait += 1
            if wait % config.train.optimizer.lr_patience == 0:
                logger.warning("Performance has not been improved for {} epochs, updating learning rate".format(wait))
                trainer.update_lr()
            if wait == config.train.optimizer.early_stopping:
                logger.warning("Performance has not been improved for {} epochs, stopping train with early stopping"
                               .format(wait))
                break

        if performance['micro_f1'] > best_performance[0]:
            wait = 0
            logger.info('Improve Micro-F1 {}% --> {}%'.format(best_performance[0], performance['micro_f1']))
            best_performance[0] = performance['micro_f1']
            best_epoch[0] = epoch
            with codecs.open(os.path.join(model_checkpoint, 'best_micro_f1_prediction.txt'),'w','utf-8') as f:
                for i in range(len(performance['predict_label'])):
                    f.write(" ".join(performance['predict_label'][i])+'\n')
            save_checkpoint({
                'epoch': epoch,
                'model_name': config.model.name,
                'state_dict': poka.state_dict(),
                'best_performance': best_performance,
                'optimizer': optimize.state_dict()
            }, os.path.join(model_checkpoint, 'best_micro_' + model_name))
        if performance['macro_f1'] > best_performance[1]:
            wait = 0
            logger.info('Improve Macro-F1 {}% --> {}%'.format(best_performance[1], performance['macro_f1']))
            best_performance[1] = performance['macro_f1']
            best_epoch[1] = epoch
            with codecs.open(os.path.join(model_checkpoint, 'best_macro_f1_prediction.txt'),'w','utf-8') as f:
                for i in range(len(performance['predict_label'])):
                    f.write(" ".join(performance['predict_label'][i])+'\n')
            save_checkpoint({
                'epoch': epoch,
                'model_name': config.model.name,
                'state_dict': poka.state_dict(),
                'best_performance': best_performance,
                'optimizer': optimize.state_dict()
            }, os.path.join(model_checkpoint, 'best_macro_' + model_name))

        if epoch % 10 == 1:
            save_checkpoint({
                'epoch': epoch,
                'model_name': config.model.name,
                'state_dict': poka.state_dict(),
                'best_performance': best_performance,
                'optimizer': optimize.state_dict()
            }, os.path.join(model_checkpoint, model_name + '_epoch_' + str(epoch)))

        logger.info('Epoch {} Time Cost {} secs.'.format(epoch, time.time() - start_time))
        if hasattr(torch.cuda, 'empty_cache'):
    	    torch.cuda.empty_cache()
    
    best_epoch_model_file = os.path.join(model_checkpoint, 'best_micro_' + model_name)
    if os.path.isfile(best_epoch_model_file):
        load_checkpoint(best_epoch_model_file, model=poka,
                        config=config,
                        optimizer=optimize)
        trainer.eval(test_loader, best_epoch[0], 'TEST')

    best_epoch_model_file = os.path.join(model_checkpoint, 'best_macro_' + model_name)
    if os.path.isfile(best_epoch_model_file):
        load_checkpoint(best_epoch_model_file, model=poka,
                        config=config,
                        optimizer=optimize)
        trainer.eval(test_loader, best_epoch[1], 'TEST')

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

    train(configs)
