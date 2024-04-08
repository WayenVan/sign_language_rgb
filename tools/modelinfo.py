#! /usr/bin/env python3
from hydra.utils import instantiate
from hydra import compose, initialize_config_dir
import click
import os
from omegaconf import OmegaConf
import torch.nn as nn
import torchinfo
import sys
sys.path.append('src')

@click.command()
@click.option('--config-dir', default='configs')
@click.option('-cn', default='run/train/vitpose_conformer_ddp')
@click.option('-d', '--depth', default=3)
@click.option('--device', default='cuda')
def main(config_dir, cn, depth, device):
    path = os.path.abspath(config_dir)
    initialize_config_dir(path)
    cfg = compose(cn)
    dataset = instantiate(cfg.data.dataset.val)
    loader = instantiate(cfg.data.loader.val, dataset=dataset, batch_size=1)
    vocab = loader.dataset.vocab
    model: nn.Module = instantiate(cfg.model, vocab=vocab).to(device)

    data = next(iter(loader))
    video = data['video'].to(device)
    gloss = data['gloss'].to(device)
    video_length = data['video_length'].to(device)
    
    torchinfo.summary(model, input_data=(video, video_length), depth=depth)

if __name__ == '__main__':
    main()
    

