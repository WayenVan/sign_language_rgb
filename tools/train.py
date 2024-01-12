from omegaconf import OmegaConf, DictConfig
import sys
import logging
sys.path.append('src')
from hydra.utils import instantiate

import torch
from torch.utils.data.dataloader import DataLoader
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from csi_sign_language.engines.trainers import Trainner
import hydra
import os
import shutil

logger = logging.getLogger('main')

@hydra.main(version_base=None, config_path='../configs/train', config_name='default.yaml')
def main(cfg: DictConfig):
    script = os.path.abspath(__file__)
    save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    shutil.copyfile(script, os.path.join(save_dir, 'script.py'))
    logger.info('building model and dataloaders')
    
    train_loader: DataLoader = instantiate(cfg.data.train_loader)
    val_loader: DataLoader = instantiate(cfg.data.val_loader)
    
    
    model: Module = instantiate(cfg.model)
    opt: Optimizer = instantiate(cfg.optimizer, model.parameters())

    logger.info('building trainner and inferencer')
    trainer: Trainner = instantiate(cfg.trainner, vocab=train_loader.dataset.vocab, logger=logger)
    logger.info('training loop start')
    best_wer_value = 1000
    for epoch in range(cfg.epoch):
        logger.info(f'epoch {epoch}')
        wer_value = trainer.do_train(model, train_loader, val_loader, opt)
        if wer_value < best_wer_value:
            best_wer_value = wer_value
            torch.save(model, os.path.join(save_dir, 'model.pth'))
            logger.info(f'best value saved')
        logger.info(f'finish one epoch')
        
        
if __name__ == '__main__':
    main()