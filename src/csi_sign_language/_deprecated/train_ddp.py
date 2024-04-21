#! /usr/bin/env python3

from omegaconf import OmegaConf, DictConfig
import time
import sys
import logging
sys.path.append('src')
from hydra.utils import instantiate
import uuid
import torch
from torch.utils.data.dataloader import DataLoader
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from csi_sign_language.engines.trainner import Trainner
from csi_sign_language.engines.inferencer import Inferencer
from csi_sign_language.evaluation.ph14.post_process import post_process
from csi_sign_language.data_utils.ph14.wer_evaluation_python import wer_calculation
from csi_sign_language.utils.misc import is_debugging, info, warn, clean
from csi_sign_language.utils.git import save_git_diff_to_file, get_current_git_hash, save_git_hash
import hydra
import os
import shutil
from torch import nn
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from datetime import datetime
logger = logging.getLogger('main')
local_rank = int(os.getenv('LOCAL_RANK'))
device = f'cuda:{local_rank}'
assert local_rank is not None

@hydra.main(version_base='1.3.2', config_path='../configs', config_name='run/train/vitpose_conformer_ddp')
def main(cfg: DictConfig):
    snap = load_snap(cfg.snap_path)

    global logger
    setup()

    is_main_rank = (local_rank == 0)
    print(is_main_rank)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)

    if is_main_rank:
        current_time = datetime.now()
        file_name = os.path.basename(__file__)
        save_dir = os.path.join('outputs', file_name[:-3], current_time.strftime("%Y-%m-%d_%H-%M-%S"))
        if snap is not None:
            save_dir = snap['save_dir']

        os.makedirs(save_dir, exist_ok=True)
        if is_debugging():
            with open(os.path.join(save_dir, 'debug'), 'w'):
                pass
        info(logger, 'saving git info')
        save_git_hash(os.path.join(save_dir, 'git_version.bash'))
        save_git_diff_to_file(os.path.join(save_dir, 'changes.patch'))
        _setup_logger(logger, save_dir)
        
        #save config
        cfg_string = OmegaConf.to_yaml(cfg)
        with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
            OmegaConf.save(cfg, f)
    else:
        logger = None
        f = open(os.devnull, 'w')
        sys.stdout = f

    info(logger, 'building model and dataloaders')
    
    #initialize data 
    model, loss_fn, train_loader, val_loader, vocab = build_model_and_data(cfg)

    #initialize record list
    metas = []
    train_id = uuid.uuid1()

    #load checkpoint
    if cfg.load_weights:
        info(logger, 'loading checkpoint')
        metas = load_checkpoints(cfg, model.module)
        _log_history(metas, logger)

    #!important, this train will set the parameter states in the model.
    model.train()
    opt, lr_scheduler, trainer, inferencer = build_engines(cfg, model)
    
    # load snapshot if it exist
    pre_epoch = 0
    if snap is not None:
        pre_epoch, meta = set_snap(snap, model.module, opt, lr_scheduler)

    best_wer_value = metas[-1]['val_wer'] if len(metas)>0 else 1000.
    for real_epoch in range(pre_epoch, cfg.epoch):
        warn(logger, f'load snap is {(snap is not None)}')
        clean() 
        #train
        train_loader.sampler.set_epoch(real_epoch)

        lr = lr_scheduler.get_last_lr()
        info(logger, f'epoch {real_epoch}, lr={lr}')

        start_time = time.time()
        mean_loss, hyp_train, gt_train= trainer.do_train(model, loss_fn, train_loader, opt, getattr(cfg, 'data_excluded', []))
        train_time = time.time() - start_time

        train_wer = wer_calculation(gt_train, hyp_train)
        info(logger, f'training finished, mean loss: {mean_loss}, wer: {train_wer}, total time: {train_time}')

        lr_scheduler.step()
        info(logger, f'finish one epoch')

        #validation
        if is_main_rank:
            ids, hypothesis, ground_truth = inferencer.do_inference(model.module, val_loader)
            hypothesis = post_process(hypothesis)
            val_wer = wer_calculation(ground_truth, hypothesis)
            info(logger, f'validation finished, wer: {val_wer}')

            #save essential informations 
            metas.append(dict(
                train_wer=train_wer,
                val_wer=val_wer,
                lr = lr,
                train_loss=mean_loss.item(),
                epoch=real_epoch,
                train_time=train_time,
                train_id=train_id
            ))
            
            
            if val_wer < best_wer_value:
                best_wer_value = val_wer
                torch.save({
                    'model_state': model.module.state_dict(),
                    'meta': metas,
                    'cfg': cfg_string,
                    }, os.path.join(save_dir, 'checkpoint.pt'))
                info(logger, f'best checkpoint saved')

            save_snap(cfg.snap_path, model.module, opt, lr_scheduler, real_epoch, save_dir, metas)
        clean()

    
    cleanup()

def setup():
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("nccl")

def cleanup():
    dist.destroy_process_group()

def save_snap(path, model, opt, lr, epoch, save_dir, metas):
    os.makedirs(path, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': opt.state_dict(),
        'lr': lr.state_dict(),
        'save_dir': save_dir,
        'meta': metas
    }, os.path.join(path, 'snap.pt'))
    
def set_snap(snap, model: nn.Module, opt: Optimizer, lr: LRScheduler):
    model.load_state_dict(snap['state_dict'])
    opt.load_state_dict(snap['optimizer'])
    lr.load_state_dict(snap['lr'])
    return snap['epoch'] + 1, snap['meta']

def load_snap(path):
    file_path = os.path.join(path, 'snap.pt')
    if os.path.exists(file_path):
        warn(logger, '----------------find snap loading--------------------')
        time.sleep(3)
        snap = torch.load(file_path, map_location=device)
    else:
        snap = None
    return snap


def build_model_and_data(cfg):
    #initialize data 
    train_set = instantiate(cfg.data.dataset.train)
    val_set = instantiate(cfg.data.dataset.val)
    vocab = train_set.vocab
    
    train_sampler = DistributedSampler(train_set)
    train_loader: DataLoader = instantiate(cfg.data.loader.train, dataset=train_set, sampler=train_sampler)
    val_loader: DataLoader = instantiate(cfg.data.loader.val, dataset=val_set)

    #initialize trainning essential
    model: Module = instantiate(cfg.model, vocab=vocab).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(
        model, 
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True
        )
    
    loss_fn: nn.Module = instantiate(cfg.loss, device=device)
    return model, loss_fn, train_loader, val_loader, vocab

def load_checkpoints(cfg, model):
    info(logger, 'loading checkpoint')
    checkpoint = torch.load(cfg.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    metas = checkpoint['meta']
    return metas

def build_engines(cfg, model):
    opt: Optimizer = instantiate(cfg.optimizer, filter(lambda p: p.requires_grad, model.parameters()))
    lr_scheduler: LRScheduler = instantiate(cfg.lr_scheduler, opt)
    trainer: Trainner = instantiate(cfg.engines.trainner, logger=logger, device=device)
    inferencer: Inferencer = instantiate(cfg.engines.inferencer, logger=logger, device=device) 
    return opt, lr_scheduler, trainer, inferencer
    

def _log_history(metas, logger: logging.Logger):
    info(logger, '-----------showing training history--------------')
    for meta in metas:
        info(logger, f"train id: {meta['train_id']}")
        info(logger, f"epoch: {meta['epoch']}")
        info(logger, "lr: {}, train loss: {}, train wer: {}, val wer: {}".format(meta['lr'], meta['train_loss'], meta['train_wer'], meta['val_wer']))
    info(logger, '-----------finish history------------------------')
    
def _setup_logger(l, save_dir):
    handler = logging.FileHandler(os.path.join(save_dir, 'train.log'))
    l.addHandler(handler)

    
if __name__ == '__main__':
    main()