import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from logging import Logger
from tqdm import tqdm
from ..models.model import *
from einops import rearrange
import numpy as np
from ..utils.inspect import *
from ..evaluation.wer import wer
from ..utils.decode import CTCDecoder
from ..utils.data import *
from typing import *
from ..modules.loss import GlobalLoss

class Trainner():
    
    def __init__(
        self,
        device,
        logger,
        message_interval,
        ) -> None:

        self.device = device
        self.message_interval = message_interval
        self.logger: logging.Logger = logger.getChild(__class__.__name__)
        if self.device == 'cuda':
            self.scaler = torch.cuda.amp.grad_scaler.GradScaler()
        
    def do_train(self, model, train_loader, opt, non_blocking=False):
        model.to(self.device)
        model.train()
        self.logger.info('start training')
        losses = []
        hyp = []
        gt = []
        for idx, data in enumerate(tqdm(train_loader)):
            opt.zero_grad()
            video = data['video'].to(self.device, non_blocking=non_blocking)
            video = rearrange(video, 'n t c h w -> t n c h w') #batch first
            gloss = data['gloss'].to(self.device, non_blocking=non_blocking)
            video_length: torch.Tensor = data['video_length'].to(self.device)
            gloss_length: torch.Tensor = data['gloss_length'].to(self.device)
            
            if 'cuda' in self.device:
                with torch.autocast('cuda'):
                    outputs = model(video, video_length)
                    loss = model.criterion(outputs, gloss, gloss_length)
            else:
                    outputs = model(video, video_length)
                    loss = model.criterion(outputs)
            
            #remove nan:
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warn(f"loss is {loss.item()}")
                self.logger.warn(f"data_id {data['id']}")
                continue
            
            if self.device == 'cuda':
                self.scaler.scale(loss).backward()
                self.scaler.step(opt)
                self.scaler.update()
            else:
                loss.backward()
                opt.step()
                
            if self.message_interval != -1 and idx % self.message_interval == 0:
                self.logger.info(f'iteration index: {idx}, batch loss: {loss.item()}')
            
            losses.append(loss.item())
            hyp += outputs['seq_out_label']
            gt += data['gloss_label']
            
        return np.mean(losses), hyp, gt

            
        
        


            


