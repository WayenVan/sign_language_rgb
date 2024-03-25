import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from logging import Logger
from tqdm import tqdm
import numpy as np
from ..utils.inspect import *
from ..utils.data import *
from typing import *

class Trainner():
    
    def __init__(
        self,
        device,
        logger,
        message_interval,
        use_amp=True
        ) -> None:

        self.device = device
        self.message_interval = message_interval
        self.use_amp = use_amp
        self.logger: logging.Logger = logger.getChild(__class__.__name__)

        if self.device == 'cuda':
            self.scaler = torch.cuda.amp.grad_scaler.GradScaler()
        
    def do_train(self, model, train_loader, opt, non_blocking=False, data_excluded=None):
        model.to(self.device)
        model.train()
        self.logger.info('start training')
        losses = []
        hyp = []
        gt = []
        for idx, data in enumerate(tqdm(train_loader)):
            opt.zero_grad()

            #remove bad data
            if data_excluded != None:
                if any(id in data_excluded for id in data['id']):
                    self.logger.warn(f"data excluded: {data['id']}")
                    del data
                    continue

            video = data['video'].to(self.device, non_blocking=non_blocking)
            gloss = data['gloss'].to(self.device, non_blocking=non_blocking)
            video_length: torch.Tensor = data['video_length'].to(self.device)
            gloss_length: torch.Tensor = data['gloss_length'].to(self.device)
            
            if 'cuda' in self.device and self.use_amp:
                with torch.autocast('cuda'):
                    outputs = model(video, video_length)
                    loss = model.criterion(outputs, gloss, gloss_length)
            else:
                    outputs = model(video, video_length)
                    loss = model.criterion(outputs, gloss, gloss_length)
            
            #remove nan:
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warn(f"loss is {loss.item()}")
                self.logger.warn(f"data_id {data['id']}")
                #clear calculation graph
                del data
                del loss
                continue
            
            if self.device == 'cuda' and self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(opt)
                self.scaler.update()
            else:
                loss.backward()
                opt.step()
                
            if self.message_interval != -1 and idx % self.message_interval == 0:
                self.logger.info(f"max memory: {torch.cuda.max_memory_allocated()}, memory: {torch.cuda.memory_allocated()}")
                self.logger.info(f'iteration index: {idx}, batch loss: {loss.item()}')
            
            losses.append(loss.item())
            hyp += outputs.label
            gt += data['gloss_label']
        
        opt.zero_grad()
            
        return np.mean(losses), hyp, gt

            
        
        


            


