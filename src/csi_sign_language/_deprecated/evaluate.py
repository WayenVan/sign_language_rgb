#! /usr/bin/env python3
from omegaconf import OmegaConf, DictConfig
from torch import nn
import sys
import logging
sys.path.append('src')
from hydra.utils import instantiate
from matplotlib import pyplot as plt
import torch
from csi_sign_language.engines.inferencer import Inferencer
from csi_sign_language.evaluation.ph14.post_process import post_process
from csi_sign_language.evaluation.ph14.wer_evaluation_sclite import eval
from csi_sign_language.data_utils.ph14.wer_evaluation_python import wer_calculation
import hydra
import os
import json

from tensorboardX import SummaryWriter

logger = logging.getLogger('main')

@hydra.main(version_base=None, config_path='../configs/evaluate', config_name='default.yaml')
def main(cfg: DictConfig):
    work_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    train_cfg = OmegaConf.load(cfg.train_config)

    test_loader = instantiate(cfg.data.test_loader)
    vocab = test_loader.dataset.get_vocab()
    
    model: torch.nn.Module = instantiate(train_cfg.model, vocab=vocab)
    model.to(cfg.device)
    checkpoint = torch.load(cfg.checkpoint)
    model.load_state_dict(checkpoint['model_state'])

    #tensorboard
    tbx_dir = os.path.join(work_dir, 'tensorboard')
    os.mkdir(tbx_dir)
    writer = SummaryWriter(tbx_dir)
    _tbx_write_graph(writer, model, test_loader)
    
    #inference
    inferencer: Inferencer = instantiate(cfg.inferencer, logger=logger)
    ids, hypothesis, ground_truth = inferencer.do_inference(model, test_loader)
    
    
    #wer calculated by python
    print(wer_calculation(ground_truth, post_process(hypothesis)))
    
    #better detail provided by sclite. need to merge for better performance
    wer_value = eval(ids, work_dir, post_process(hypothesis, regex=False, merge=True), test_loader.dataset.get_stm(), 'hyp.ctm', cfg.evaluation_tool)
    print(wer_value[0])
    
    #evaluation of each result
    wers_every = [wer_calculation([gt], [hypo]) for hypo, gt in zip(hypothesis, ground_truth)]
    ret = []
    for id, wer, gt, pre, post in list(zip(ids, wers_every, ground_truth, hypothesis, post_process(hypothesis))):
        ret.append(dict(
            id=id,
            wer=wer,
            gt=gt,
            pre=pre,
            post=post
        ))
    with open(os.path.join(work_dir, 'result.json'), 'w') as f:
        json.dump(ret, f, indent=4)
    
    
    
def _tbx_write_graph(writer: SummaryWriter, model: nn.Module, loader):
    device = next(iter(model.parameters())).device
    data = next(iter(loader))
    video = data['video'].to(device)
    video_length = data['video_length'].to(device)
    writer.add_graph(model.backbone, (video, video_length))
        
    
    
if __name__ == '__main__':
    main()