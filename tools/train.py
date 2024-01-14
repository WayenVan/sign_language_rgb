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
from csi_sign_language.engines.trainers import Trainner
from csi_sign_language.engines.inferencers import Inferencer
from csi_sign_language.utils.data import flatten_concatenation
from csi_sign_language.utils.metrics import wer
import hydra
import os
import shutil
from itertools import chain
logger = logging.getLogger('main')
import lmdb

@hydra.main(version_base=None, config_path='../configs/train', config_name='default.yaml')
def main(cfg: DictConfig):
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    env = lmdb.open(os.path.join(cfg.phoenix14_root, cfg.data.subset, 'feature_database'))
    script = os.path.abspath(__file__)
    save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    shutil.copyfile(script, os.path.join(save_dir, 'script.py'))
    logger.info('building model and dataloaders')
    
    train_loader: DataLoader = instantiate(cfg.data.train_loader, dataset={'lmdb_env': env})
    val_loader: DataLoader = instantiate(cfg.data.val_loader, dataset={'lmdb_env': env})
    vocab = train_loader.dataset.vocab
    
    model: Module = instantiate(cfg.model)
    opt: Optimizer = instantiate(cfg.optimizer, model.parameters())
    lr_scheduler: LambdaLR = instantiate(cfg.lr_scheduler, opt)
    
    logger.info('building trainner and inferencer')
    trainer: Trainner = instantiate(cfg.trainner, vocab=vocab, logger=logger)
    inferencer: Inferencer = instantiate(cfg.inferencer, vocab=vocab, logger=logger) 
    logger.info('training loop start')
    best_wer_value = 1000
    for epoch in range(cfg.epoch):
        logger.info(f'epoch {epoch}')
        mean_loss = trainer.do_train(model, train_loader, opt)
        logger.info(f'training finished, mean loss: {mean_loss}')
        hypothesis, ground_truth = inferencer.do_inference(model, val_loader)
        wer_value = wer(ground_truth, hypothesis)
        logger.info(f'validation finished, wer: {wer_value}')
        if wer_value < best_wer_value:
            best_wer_value = wer_value
            torch.save(model, os.path.join(save_dir, 'model.pth'))
            logger.info(f'best model saved')
        lr_scheduler.step()
        logger.info(f'finish one epoch')
        
        
if __name__ == '__main__':
    main()