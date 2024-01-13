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
from csi_sign_language.engines.inferencers import Inferencer
from csi_sign_language.utils.data import flatten_concatenation
from csi_sign_language.utils.metrics import wer
import hydra
import os
import shutil
from itertools import chain
logger = logging.getLogger('main')

@hydra.main(version_base=None, config_path='../configs/evaluate', config_name='default.yaml')
def main(cfg: DictConfig):
    model = torch.load(cfg.model_path)
    test_loader = instantiate(cfg.data.test_loader)
    vocab = test_loader.dataset. get_vocab()
    inferencer: Inferencer = instantiate(cfg.inferencer, logger=logger, vocab=vocab)
    hypothesis, ground_truth = inferencer.do_inference(model, test_loader)
    wer_value = wer(hypothesis, ground_truth)
    logger.info(f'validation finished, wer: {wer_value}')
    

if __name__ == '__main__':
    main()