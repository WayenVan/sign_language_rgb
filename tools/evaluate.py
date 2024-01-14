from omegaconf import OmegaConf, DictConfig
import sys
import logging
sys.path.append('src')
from hydra.utils import instantiate

import torch
from csi_sign_language.engines.inferencers import Inferencer
from csi_sign_language.utils.metrics import wer, wer_mean
import hydra
import os

logger = logging.getLogger('main')

@hydra.main(version_base=None, config_path='../configs/evaluate', config_name='default.yaml')
def main(cfg: DictConfig):
    result = OmegaConf.create()
    save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    model = torch.load(cfg.model_path)
    test_loader = instantiate(cfg.data.test_loader)
    vocab = test_loader.dataset. get_vocab()
    inferencer: Inferencer = instantiate(cfg.inferencer, logger=logger, vocab=vocab)
    hypothesis, ground_truth = inferencer.do_inference(model, test_loader)
    wer_value = wer(ground_truth, hypothesis)
    logger.info(f'validation finished, wer: {wer_value}')
    
    result.wer = wer_value
    result.results = [{'hypothesis': h, 'ground_truth': gt} for h, gt in zip(hypothesis, ground_truth)]
    OmegaConf.save(result, os.path.join(save_dir, 'result.yaml'))

if __name__ == '__main__':
    main()