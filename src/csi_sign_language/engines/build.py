from omegaconf import OmegaConf, DictConfig
from .trainers import Trainner
from .inferencers import *
import numpy as np
import os
from csi_sign_language.utils.logger import strtime

def build_trainner(
    cfg: DictConfig,
    logger
):
    save_dir = os.path.join(cfg.save_dir, f'experiment-{strtime()}')
    return Trainner(
        cfg.data.num_class,
        cfg.data.vocab_dir,
        save_dir,
        cfg.device,
        logger,
        debug=cfg.debug
    )

def build_inferencer(
    cfg: DictConfig
):
    pass