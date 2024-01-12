import sys
sys.path.append('src')
import os
from pathlib import Path
from csi_sign_language.data.dataset.phoenix14 import Phoenix14Dataset
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

def main():
    cfg = OmegaConf.load('configs/train/default.yaml')
    cfg2 = OmegaConf.load('/home/jingyan/Documents/sign_language_rgb/outputs/2024-01-11/13-41-00/.hydra/config.yaml')
    loader = hydra.utils.instantiate(cfg.data.train_loader)
    loader2 = hydra.utils.instantiate(cfg2.train_loader)
    for i in tqdm(loader):
        pass
        
    for i in tqdm(loader2):
        pass


main()