#! /usr/bin/env python3

from omegaconf import OmegaConf, DictConfig
import time
import sys
import logging
sys.path.append('src')
from hydra.utils import instantiate
import uuid
import torch
from torch.utils.data.dataloader import DataLoader
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from csi_sign_language.engines.trainner import Trainner
from csi_sign_language.engines.inferencer import Inferencer
from csi_sign_language.evaluation.ph14.post_process import post_process
from csi_sign_language.data_utils.ph14.wer_evaluation_python import wer_calculation
from csi_sign_language.utils.misc import is_debugging
from csi_sign_language.utils.git import save_git_diff_to_file, get_current_git_hash, save_git_hash
import hydra
import os
logger = logging.getLogger('main')
import numpy as np


@hydra.main(version_base='1.3.2', config_path='../configs', config_name='run/train/resnet_trans')
def main(cfg: DictConfig):
    print(os.getcwd())
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    if is_debugging():
        with open(os.path.join(save_dir, 'debug'), 'w'):
            pass

    logger.info('saving git info')
    save_git_hash(os.path.join(save_dir, 'git_version.bash'))
    save_git_diff_to_file(os.path.join(save_dir, 'changes.patch'))

    logger.info('building model and dataloaders')
    
    #initialize data 
    train_loader: DataLoader = instantiate(cfg.data.loader.train)
    val_loader: DataLoader = instantiate(cfg.data.loader.val)
    vocab = train_loader.dataset.vocab
    
    #initialize trainning essential
    model: Module = instantiate(cfg.model, vocab=vocab)
    #move model before optimizer initialize
    model.to(cfg.device)

    #initialize record list
    metas = []
    last_epoch = -1
    train_id = uuid.uuid1()

    #load checkpoint
    if cfg.is_resume or cfg.load_weights:
        logger.info('loading checkpoint')
        checkpoint = torch.load(cfg.checkpoint)
        model.load_state_dict(checkpoint['model_state'])
        metas = checkpoint['meta']
        _log_history(checkpoint, logger)

    #!important, this train will set the parameter states in the model.
    model.train()
    opt: Optimizer = instantiate(cfg.optimizer, filter(lambda p: p.requires_grad, model.parameters()))
    
    if cfg.is_resume:
        last_epoch = metas[-1]['epoch']
        opt.load_state_dict(checkpoint['optimizer_state'])
        train_id = metas[-1]['train_id']

    
    lr_scheduler: LRScheduler = instantiate(cfg.lr_scheduler, opt, last_epoch=last_epoch)
    
    logger.info('building trainner and inferencer')
    trainer: Trainner = instantiate(cfg.engines.trainner, logger=logger)
    inferencer: Inferencer = instantiate(cfg.engines.inferencer, logger=logger) 
    logger.info('training loop start')
     
    best_wer_value = metas[-1]['val_wer'] if len(metas) > 0 else 1000.
    for i in range(cfg.epoch):
        real_epoch = last_epoch + i + 1

        #train
        lr = lr_scheduler.get_last_lr()
        logger.info(f'epoch {real_epoch}, lr={lr}')

        start_time = time.time()
        mean_loss, hyp_train, gt_train= trainer.do_train(model, train_loader, opt, getattr(cfg, 'data_excluded', []))
        train_time = time.time() - start_time
        train_wer = wer_calculation(gt_train, hyp_train)
        logger.info(f'training finished, mean loss: {mean_loss}, wer: {train_wer}, total time: {train_time}')

        #validation
        ids, hypothesis, ground_truth = inferencer.do_inference(model, val_loader)
        hypothesis = post_process(hypothesis)
        val_wer = wer_calculation(ground_truth, hypothesis)
        logger.info(f'validation finished, wer: {val_wer}')

        lr_scheduler.step()
        logger.info(f'finish one epoch')
        
        #save essential informations 
        metas.append(dict(
            train_wer=train_wer,
            val_wer=val_wer,
            lr = lr,
            train_loss=mean_loss.item(),
            epoch=real_epoch,
            train_time=train_time,
            train_id=train_id
        ))
        
        
        if val_wer < best_wer_value:
            best_wer_value = val_wer
            torch.save({
                'model_state': model.cpu().state_dict(),
                'optimizer_state': opt.state_dict(),
                'meta': metas
                }, os.path.join(save_dir, 'checkpoint.pt'))
            logger.info(f'best checkpoint saved')


def _log_history(checkpoint, logger: logging.Logger):
    logger.info('-----------showing training history--------------')
    for info in checkpoint['meta']:
        logger.info(f"train id: {info['train_id']}")
        logger.info(f"epoch: {info['epoch']}")
        logger.info("lr: {}, train loss: {}, train wer: {}, val wer: {}".format(info['lr'], info['train_loss'], info['train_wer'], info['val_wer']))
    logger.info('-----------finish history------------------------')

    
if __name__ == '__main__':
    main()