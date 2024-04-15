
#! /usr/bin/env python3
from omegaconf import OmegaConf, DictConfig
from torch import nn
import sys
import logging
sys.path.append('src')
from hydra.utils import instantiate
from matplotlib import pyplot as plt
import torch
from csi_sign_language.data_utils.ph14.evaluator_sclite import Pheonix14Evaluator
from csi_sign_language.data_utils.ph14.post_process import post_process

from csi_sign_language.models.slr_model import SLRModel
import hydra
import os
import json
from datetime import datetime
import click
from torchmetrics.text import WordErrorRate

from lightning.pytorch.trainer import Trainer

@click.option('--config', '-c', default='outputs/train_lightning/2024-04-12_23-37-41/config.yaml')
@click.option('-ckpt', '--checkpoint', default='outputs/train_lightning/2024-04-12_23-37-41/epoch=32_wer-val=28.43_lr=4.00e-05_loss=3.81.ckpt')
@click.option('--ph14_root', default='dataset/phoenix2014-release')
@click.command()
def main(config, checkpoint, ph14_root):
    current_time = datetime.now()
    file_name = os.path.basename(__file__)
    save_dir = os.path.join('outputs', file_name[:-3], current_time.strftime("%Y-%m-%d_%H-%M-%S"))
    cfg = OmegaConf.load(config)
    
    dataset = hydra.utils.instantiate(cfg.data.dataset.test)
    dataloader = hydra.utils.instantiate(cfg.data.loader.test, dataset)
    model = SLRModel.load_from_checkpoint(checkpoint, cfg=cfg)
    t = Trainer(
        accelerator='gpu',
        strategy='ddp',
        devices=2,
        logger=False,
        enable_checkpointing=False,
        precision='32-true',
    )
    outputs = t.validate(model, dataloader)[0]

    # outputs = t.predict(model, dataloader)

    # if t.local_rank == 0:
    #     os.makedirs(save_dir, exist_ok=True)
    #     results = []
    #     for output in outputs:
    #         results += output

    #     results = tuple(zip(*results))
    #     ids, hyps, gts = results[0], results[1], results[2]
    #     hyps = post_process(hyps)
    #     evaluator = Pheonix14Evaluator(ph14_root, 'multisigner')
    #     evaluator.eval(save_dir, ids, hyps, mode='test')
        
        
    
if __name__ == '__main__':
    main() 
    