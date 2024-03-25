#! /usr/bin/env python3
from hydra.utils import instantiate
import click
import os
from omegaconf import OmegaConf
import torch.nn as nn
import torchinfo
import sys
sys.path.append('src')

@click.command()
@click.option('--config-dir', default='configs/train')
@click.option('-cn', default='x3d_litflow_trans.yaml')
@click.option('-d', '--depth', default=3)
@click.option('--device', default='cuda')
def main(config_dir, cn, depth, device):
    configfile = os.path.join(config_dir, cn)
    cfg = OmegaConf.load(configfile)
    loader = instantiate(cfg.data.val_loader, batch_size=1)
    vocab = loader.dataset.vocab
    model: nn.Module = instantiate(cfg.model, vocab=vocab).to(device)

    data = next(iter(loader))
    video = data['video'].to(device)
    gloss = data['gloss'].to(device)
    video_length = data['video_length'].to(device)
    
    torchinfo.summary(model, input_data=(video, video_length), depth=depth)

if __name__ == '__main__':
    main()
    

