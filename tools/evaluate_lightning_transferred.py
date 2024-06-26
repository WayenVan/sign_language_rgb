
#! /usr/bin/env python3
from lightning import LightningModule
from omegaconf import OmegaConf, DictConfig
from torch import nn
import sys
import logging
sys.path.append('src')
from hydra.utils import instantiate
from matplotlib import pyplot as plt
import torch
from csi_sign_language.data_utils.ph14.evaluator_sclite import Pheonix14Evaluator
from csi_sign_language.data.datamodule.ph14 import Ph14DataModule

from csi_sign_language.models.slr_model import SLRModel
import hydra
import os
import json
from datetime import datetime
import click
from torchmetrics.text import WordErrorRate

from lightning.pytorch.trainer import Trainer
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import NeptuneLogger, CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import Callback
import pickle


@click.option('--config', '-c', default='outputs/train_lightning/2024-04-17_01-59-49/config.yaml')
@click.option('-ckpt', '--checkpoint', default='outputs/train_lightning/2024-04-17_01-59-49/last.ckpt')
@click.option('--ph14_root', default='dataset/phoenix2014-release')
@click.option('--ph14_lmdb_root', default='preprocessed/ph14_lmdb')
@click.option('--tmp', default='tmp')
@click.option('--mode', default='test')
@click.command()
def main(config, checkpoint, ph14_root, ph14_lmdb_root, tmp, mode):
    
    if mode not in ['val', 'test']:
        raise NotImplementedError()

    current_time = datetime.now()
    file_name = os.path.basename(__file__)
    save_dir = os.path.join('outputs', file_name[:-3], current_time.strftime("%Y-%m-%d_%H-%M-%S"))
    cfg = OmegaConf.load(config)
    
    ck = torch.load('outputs/train_x3d_trans/checkpoint.pt')
    dm = Ph14DataModule(ph14_lmdb_root, batch_size=1, num_workers=6, train_shuffle=True, val_transform=instantiate(cfg.transforms.test), test_transform=instantiate(cfg.transforms.test))
    # model = SLRModel.load_from_checkpoint(checkpoint, cfg=cfg, map_location='cpu', ctc_search_type='beam')
    model = SLRModel(cfg, dm.train_set.get_vocab(), 'greedy')
    model.load_state_dict(ck['model_state'], strict=False)
    model.set_post_process(dm.get_post_process())

    t = Trainer(
        accelerator='gpu',
        strategy='ddp',
        devices=2,
        logger=False,
        enable_checkpointing=False,
        precision=32,
        use_distributed_sampler=True,
    )
    strategy = t.strategy
    strategy.connect(model)
    t.save_checkpoint('outputs/vitpose_trans_best/val-wer=24.ckpt')
    OmegaConf.save(cfg, 'outputs/vitpose_trans_best/config.yaml')
    return
    loader = dm.test_dataloader() if mode=='test' else dm.val_dataloader()
    t.validate(model, loader)
    outputs = t.predict(model, loader)

    results = []
    for output in outputs:
        results += output
    
    
    os.makedirs(tmp, exist_ok=True )
    with open(os.path.join(tmp, f'results{t.local_rank}.pkl'), 'wb') as f:
        pickle.dump(results, f)
    strategy.barrier()

    if t.local_rank == 0:
        results = []
        for rank in range(t.world_size):
            with open(os.path.join(tmp, f'results{rank}.pkl'), 'rb') as f:
                results += pickle.load(f)

        os.makedirs(save_dir, exist_ok=True)
        results = tuple(zip(*results))
        ids, hyps, gts = results[0], results[1], results[2]
        hyps, _= dm.get_post_process().process(hyps, gts)
        # from csi_sign_language.data_utils.ph14.post_process import post_process
        # from csi_sign_language.data_utils.ph14.wer_evaluation_python import wer_calculation
        # print(wer_calculation(gts, post_process(hyps)))

        # # hyps = post_process(hyps)

        
        evaluator = Pheonix14Evaluator(ph14_root, 'multisigner')
        evaluator.eval(save_dir, ids, hyps, mode='test' if mode == 'test' else 'dev')
        
        
    
if __name__ == '__main__':
    main() 
    