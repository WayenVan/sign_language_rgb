from omegaconf import OmegaConf, DictConfig
import sys
import logging
sys.path.append('src')
from hydra.utils import instantiate

import torch
from torch.utils.data.dataloader import DataLoader
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from csi_sign_language.engines.trainner import Trainner
from csi_sign_language.engines.inferencer import Inferencer
from csi_sign_language.utils.data import flatten_concatenation, list2vocab
from csi_sign_language.utils.metrics import wer

import hydra
import os
import shutil
from itertools import chain
logger = logging.getLogger('main')
import numpy as np
import json


@hydra.main(version_base=None, config_path='../configs/train', config_name='default_wds.yaml')
def main(cfg: DictConfig):
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    # env = lmdb.open(os.path.join(cfg.phoenix14_root, cfg.data.subset, 'feature_database'))
    script = os.path.abspath(__file__)
    save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    shutil.copyfile(script, os.path.join(save_dir, 'script.py'))
    logger.info('building model and dataloaders')
    
    #initialize data 
    transform = instantiate(cfg.data.transform)
    decoder = instantiate(cfg.data.decoder)
    
    train_set = instantiate(cfg.data.train_set).shuffle(cfg.data.shuffle).map(decoder).map(transform)
    val_set = instantiate(cfg.data.val_set).map(decoder).map(transform)
    
    train_loader: DataLoader = instantiate(cfg.data.train_loader, dataset=train_set)
    val_loader: DataLoader = instantiate(cfg.data.val_loader, dataset=val_set)
    
    with open(os.path.join(cfg.data_root, cfg.data.subset, 'info.json'), 'r') as f:
        info = json.load(f)
        
    vocab = list2vocab(info['vocab'])
    
    #initialize trainning essential
    model: Module = instantiate(cfg.model)
    opt: Optimizer = instantiate(cfg.optimizer, model.parameters())
    wer_values = []
    losses = []
    last_epoch = -1
    
    #load checkpoint
    if cfg.is_resume:
        logger.info('loading checkpoint')
        checkpoint = torch.load(cfg.checkpoint)
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state'])
        opt.load_state_dict(checkpoint['optimizer_state'])
        wer_values = checkpoint['wer']
        losses = checkpoint['loss']
    
    
    lr_scheduler: LambdaLR = instantiate(
        cfg.lr_scheduler, 
        opt,
        last_epoch=last_epoch)
    
    logger.info('building trainner and inferencer')
    trainer: Trainner = instantiate(cfg.trainner, vocab=vocab, logger=logger)
    inferencer: Inferencer = instantiate(cfg.inferencer, vocab=vocab, logger=logger) 
    logger.info('training loop start')
    best_wer_value = 1000
    for i in range(cfg.epoch):
        real_epoch = last_epoch + i + 1
            
        logger.info(f'epoch {real_epoch}')
        mean_loss = trainer.do_train(model, train_loader, opt, non_blocking=True)
        # mean_loss = np.array([0.])
        logger.info(f'training finished, mean loss: {mean_loss}')
        hypothesis, ground_truth = inferencer.do_inference(model, val_loader)
        wer_value = wer(ground_truth, hypothesis)
        logger.info(f'validation finished, wer: {wer_value}')
        
        wer_values.append(wer_value)
        losses.append(mean_loss.item())
        if wer_value < best_wer_value:
            best_wer_value = wer_value
            torch.save({
                'epoch': real_epoch,
                'model_state': model.state_dict(),
                'optimizer_state': opt.state_dict(),
                'wer': wer_values,
                'loss': losses
                }, os.path.join(save_dir, 'checkpoint.pt'))
            logger.info(f'best checkpoint saved')
        lr_scheduler.step()
        logger.info(f'finish one epoch')


if __name__ == '__main__':
    main()