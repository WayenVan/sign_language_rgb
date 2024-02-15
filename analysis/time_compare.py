import torch 
import copy
import sys
import tqdm
sys.path.append('src')
import logging

from omegaconf import OmegaConf
from hydra.utils import instantiate
import re
import time
from functools import partial

from torch.utils.data.dataloader import DataLoader
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from csi_sign_language.engines.trainner import Trainner
from csi_sign_language.engines.inferencer import Inferencer
from csi_sign_language.utils.data import flatten_concatenation
from csi_sign_language.evaluation.ph14.post_process import post_process
from csi_sign_language.evaluation.ph14.wer_evaluation_python import wer_calculation
def pre_forward(m, input, name, storage):
    storage[name] = {}
    storage[name]['start'] = time.time()
    
def forward(m, input, output, name, storage):
    storage[name]['end'] = time.time()
    storage[name]['delta'] = storage[name]['end'] - storage[name]['start']

def hrnet_rnn():
    cfg = OmegaConf.load("configs/train/hrnet_rnn.yaml")
    dataloader = instantiate(cfg.data.train_loader)
    model: torch.nn.Module = instantiate(cfg.model, vocab=dataloader.dataset.get_vocab()).cuda()

    storage = {}
        

    for name, m in model.named_modules():
        # print(name)
        if re.match(r'backbone\.[a-z\_]+$', name) or \
            re.match(r'backbone\.sthrnet\.[a-z\_]+$', name) or \
            re.match(r'backbone\.sthrnet\.lrnet\.[a-z\_0-9]+$', name):
            m.register_forward_pre_hook(partial(pre_forward, name=name, storage=storage))
            m.register_forward_hook(partial(forward, name=name, storage=storage))


    data = next(iter(dataloader))
    with torch.autocast('cuda'):
        video = data['video'].cuda()
        lgt = data['video_length'].cuda()
        model(video, lgt)

    for key in storage.keys():
        print(f"{key} value: {storage[key]['delta']: E}")

def resnet():

    cfg = OmegaConf.load("configs/train/default.yaml")

    dataloader = instantiate(cfg.data.train_loader)
    model: torch.nn.Module = instantiate(cfg.model, vocab=dataloader.dataset.get_vocab()).cuda()

    storage = {}
        

    for name, m in model.named_modules():
        if re.match(r'backbone.[a-z\_]+$', name):
            m.register_forward_pre_hook(partial(pre_forward, name=name, storage=storage))
            m.register_forward_hook(partial(forward, name=name, storage=storage))


    data = next(iter(dataloader))
    with torch.autocast('cuda'):
        video = data['video'].cuda()
        lgt = data['video_length'].cuda()
        model(video, lgt)

    for key in storage.keys():
        print(f"{key} value: {storage[key]['delta']: E}")

def x3d():
    cfg = OmegaConf.load("configs/train/x3d_lstm.yaml")
    dataloader = instantiate(cfg.data.train_loader)
    model: torch.nn.Module = instantiate(cfg.model, vocab=dataloader.dataset.get_vocab()).cuda()
    storage = {}
        
    for name, m in model.named_modules():
        if re.match(r'backbone.[a-z\_]+$', name) or re.match(r'backbone.res_stages.[0-9]$', name):
            m.register_forward_pre_hook(partial(pre_forward, name=name, storage=storage))
            m.register_forward_hook(partial(forward, name=name, storage=storage))

    data = next(iter(dataloader))
    with torch.autocast('cuda'):
        video = data['video'].cuda()
        lgt = data['video_length'].cuda()
        model(video, lgt)

    for key in storage.keys():
        print(f"{key} value: {storage[key]['delta']: E}")

def x3d_enumerate():
    cfg = OmegaConf.load("configs/train/x3d_lstm.yaml")
    # cfg = OmegaConf.load("configs/train/default.yaml")
    train_loader = instantiate(cfg.data.train_loader)
    val_loader: DataLoader = instantiate(cfg.data.val_loader)
    model: torch.nn.Module = instantiate(cfg.model, vocab=train_loader.dataset.get_vocab()).cuda()
    storage = {}
    
    def reg():
        for name, c in model.backbone.named_modules():
            if 'x3d' in name:
                print(name)
            if re.match(r'res_stages.[0-9]$', name) or re.match(r'[a-z_]+$', name): 
                c.register_forward_pre_hook(partial(pre_forward, name=name, storage=storage))
                c.register_forward_hook(partial(forward, name=name, storage=storage))
    # for name, m in model.named_modules():
    #     if re.match(r'backbone.[a-z\_]+$', name) or re.match(r'backbone.res_stages.[0-9]$', name):
    #         m.register_forward_pre_hook(partial(pre_forward, name=name, storage=storage))
    #         m.register_forward_hook(partial(forward, name=name, storage=storage))

    logger = logging.Logger('hahaha')

    opt = instantiate(cfg.optimizer, model.parameters())
    trainer: Trainner = instantiate(cfg.trainner, logger=logger)
    trainer2 = copy.deepcopy(trainer)
    trainer2.use_amp = False
    inferencer: Inferencer = instantiate(cfg.inferencer, logger=logger) 
    model.eval()
    #ids, hypothesis, ground_truth = inferencer.do_inference(model, val_loader)
    reg()
    model.train()
    while 1:
        mean_loss, hyp_train, gt_train= trainer2.do_train(model, val_loader, opt, non_blocking=False)
        ids, hypothesis, ground_truth = inferencer.do_inference(model, val_loader)
        # for key in storage.keys():
        #     print(f"{key} value: {storage[key]['delta']: E}")
        # train_wer = wer_calculation(gt_train, hyp_train)
        # val_wer = wer_calculation(ground_truth, hypothesis)
    for key in storage.keys():
        print(f"{key} value: {storage[key]['delta']: E}")
    
if __name__ == "__main__":
    x3d_enumerate()
    # print('-----------------')
    # resnet()