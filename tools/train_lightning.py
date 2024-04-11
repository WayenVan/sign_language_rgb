#! /usr/bin/env python3

from torchtext.vocab import Vocab
from omegaconf import OmegaConf, DictConfig
import sys
sys.path.append('src')
from hydra.utils import instantiate
from torch.utils.data.dataloader import DataLoader
from csi_sign_language.utils.git import save_git_diff_to_file, get_current_git_hash, save_git_hash
from csi_sign_language.models.slr_model import SLRModel
import hydra
import os
import numpy as np

from datetime import datetime
from lightning.pytorch import seed_everything
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch import callbacks
from lightning.pytorch import trainer
from lightning.pytorch import strategies

@hydra.main(version_base='1.3.2', config_path='../configs', config_name='run/train/resnet_lstm_lightning')
def main(cfg: DictConfig):
    seed_everything(cfg.seed, workers=True)

    #set output directory
    current_time = datetime.now()
    file_name = os.path.basename(__file__)
    save_dir = os.path.join('outputs', file_name[:-3], current_time.strftime("%Y-%m-%d_%H-%M-%S"))

    csv_logger = pl_loggers.CSVLogger(save_dir, name='csv_log')
    ckpt_callback = callbacks.ModelCheckpoint(
        save_dir, 
        save_last=True, 
        filename='epoch={epoch}_wer-val={val_wer:.2f}',
        monitor='val_wer', 
        mode='min', 
        save_top_k=1,
        auto_insert_metric_name=True)

    train_loader, val_loader, vocab = build_data(cfg)
    if cfg.load_weights:
        lightning_module = SLRModel.load_from_checkpoint(cfg.checkpoint)
    else:
        lightning_module = SLRModel(cfg, vocab)
    
    t = trainer.Trainer(
        accelerator='gpu',
        # strategy=strategies.DeepSpeedStrategy(),
        strategy='ddp',
        devices=2,
        callbacks=[ckpt_callback],
        logger=[csv_logger],
        precision=16,
        log_every_n_steps=25
    )
    
    if t.local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        #save config
        with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
            OmegaConf.save(cfg, f)

        #save git
        save_git_hash(os.path.join(save_dir, 'git_version.bash'))
        save_git_diff_to_file(os.path.join(save_dir, 'changes.patch'))

    t.fit(lightning_module, train_loader, val_loader)
    return

def build_data(cfg):
    #initialize data 
    train_set = instantiate(cfg.data.dataset.train)
    val_set = instantiate(cfg.data.dataset.val)
    vocab = train_set.vocab
    
    train_loader: DataLoader = instantiate(cfg.data.loader.train, dataset=train_set)
    val_loader: DataLoader = instantiate(cfg.data.loader.val, dataset=val_set)

    return train_loader, val_loader, vocab
    
if __name__ == '__main__':
    main()