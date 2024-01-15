import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from logging import Logger
from tqdm import tqdm
from ..models.models import *
from einops import rearrange
import numpy as np
from ..utils.inspect import *
from ..utils.metrics import wer
from ..utils.decode import CTCDecoder
from ..utils.data import *
from typing import *

class Trainner():
    
    def __init__(
        self,
        num_class: int,
        vocab,
        device,
        logger,
        verbose=False
        ) -> None:

        self.NUM_CLASS = int(num_class)
        self.device = device
        self.vocab = vocab
        self.verbose=verbose
        self.loss_fn = nn.CTCLoss(reduction='mean', zero_infinity=False)
        self.logger: logging.Logger = logger.getChild(__class__.__name__)
        if self.device == 'cuda':
            self.scaler = torch.cuda.amp.grad_scaler.GradScaler()
    
        
    def do_train(self, model, train_loader, opt):
        
        model.to(self.device, non_blocking=True)
        model.train()
        self.logger.info('start training')
        losses = []
        for idx, data in enumerate(tqdm(train_loader)):
            
            opt.zero_grad()
            video = data['video'].to(self.device, non_blocking=True)
            video = rearrange(video, 'n t c h w -> t n c h w') #batch first
            gloss = data['gloss'].to(self.device, non_blocking=True)
            video_length: torch.Tensor = data['video_length'].to(self.device)
            gloss_length: torch.Tensor = data['gloss_length'].to(self.device)
            y_predict = model(video, video_length)

            loss = self.loss_fn(y_predict, gloss, video_length, gloss_length)
            if  np.isinf(loss.item()) or np.isnan(loss.item()):
                self.logger.warn(f'loss is {loss.item()}')
                self.logger.warn(f'annotation lenght {str(gloss_length)}')
                self.logger.warn(f'output size {str(y_predict.size())}')
                self.logger.warn(f'y_predict {str(y_predict)}')
                continue
            
            if self.device == 'cuda':
                self.scaler.scale(loss).backward()
                self.scaler.step(opt)
                self.scaler.update()
            else:
                loss.backward()
                opt.step()
                
            if self.verbose:
                self.logger.info(f'iteration index: {idx}, batch loss: {loss.item()}')
            
            losses.append(loss.item())
        return np.mean(losses)

            
        
        


            


