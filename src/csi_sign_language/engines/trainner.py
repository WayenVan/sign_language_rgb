import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from logging import Logger
from tqdm import tqdm
import numpy as np
from ..utils.inspect import *
from ..utils.data import *
from typing import *
from ..utils.misc import info, warn

class Trainner():
    
    def __init__(
        self,
        device,
        message_interval,
        logger=None,
        use_amp=True
        ) -> None:

        assert isinstance(device, str), 'device must be a string'
        self.device = device
        self.use_amp = use_amp
        
        if 'cuda' in self.device:
            self.is_cuda = True
        else:
            self.is_cuda = False
        
        if self.is_cuda:
            self.scaler = torch.cuda.amp.grad_scaler.GradScaler()

        self.message_interval = message_interval
        
        if logger is not None:
            self.logger: logging.Logger = logger.getChild(__class__.__name__)
        else:
            self.logger = None
        
    def do_train(self, model, loss_fn, train_loader, opt, data_excluded=None):
        model.to(self.device)
        model.train()
        losses = []
        hyp = []
        gt = []

        for idx, data in enumerate(tqdm(train_loader)):

            opt.zero_grad()

            #remove bad data
            if data_excluded != None:
                if any(id in data_excluded for id in data['id']):
                    warn(self.logger, f"data excluded: {data['id']}")
                    del data
                    continue
            
            loss, outputs = self._forward(model, loss_fn, data)

            #remove nan:
            if torch.isnan(loss) or torch.isinf(loss):
                if self.logger is not None:
                    warn(self.logger, f"loss is {loss.item()}")
                    warn(self.logger, f"data_id {data['id']}")
                #clear calculation graph
                del data
                del loss
                continue
            
            self._backward_and_update(loss, opt)

            if self.message_interval != -1 and idx % self.message_interval == 0:
                info(self.logger, f"max memory: {torch.cuda.max_memory_allocated()}, memory: {torch.cuda.memory_allocated()}")
                info(self.logger, f'iteration index: {idx}, batch loss: {loss.item()}')
            
            losses.append(loss.item())
            hyp += outputs.label
            gt += data['gloss_label']
        
        opt.zero_grad()
            
        return np.mean(losses), hyp, gt
    
    
    def _forward(self, model, loss_fn, data):
        video = data['video'].to(self.device)
        gloss = data['gloss'].to(self.device)
        video_length: torch.Tensor = data['video_length'].to(self.device)
        gloss_length: torch.Tensor = data['gloss_length'].to(self.device)
        
        if self.is_cuda and self.use_amp:
            with torch.autocast('cuda'):
                outputs = model(video, video_length)
                loss = loss_fn(outputs, video, video_length, gloss, gloss_length)
        else:
                outputs = model(video, video_length)
                loss = loss_fn(outputs, video, video_length, gloss, gloss_length)
        
        return loss, outputs
    
    def _backward_and_update(self, loss, opt):
        if self.device == 'cuda' and self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(opt)
            self.scaler.update()
        else:
            loss.backward()
            opt.step()
        

            
        
        


            


