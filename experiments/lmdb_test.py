import lmdb
import sys
sys.path.append('src')
import hydra
import omegaconf
import tqdm

cfg = omegaconf.OmegaConf.load('/home/jingyan/Documents/sign_language_rgb/configs/train/default.yaml')

loader = hydra.utils.instantiate(cfg.data.train_loader)

import pickle

print(pickle.dumps(loader))