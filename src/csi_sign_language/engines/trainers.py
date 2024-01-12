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
        debug=False
        ) -> None:

        self.NUM_CLASS = int(num_class)
        self.device = device
        self.vocab = vocab
        self.decoder = CTCDecoder(self.vocab, blank_id=0)
        self.debug=debug
        self.loss_fn = nn.CTCLoss(reduction='mean', zero_infinity=False)
        self.logger = logger.getChild(__class__.__name__)
        if self.device == 'cuda':
            self.scaler = torch.cuda.amp.grad_scaler.GradScaler()
    
        
    def _get_ground_truth(self, ground_truth, gloss_length) -> List[List[str]]:
        """

        :param ground_truth: [n t]
        :param sequence_masks: [n]
        """
        ret = []
        for batch_id, gloss in enumerate(ground_truth):
            gloss = gloss.cpu().numpy()
            l = gloss_length[batch_id].cpu().numpy()
            ret.append(self.vocab.lookup_tokens(gloss[:l]))
        return ret
        
    def do_train(self, model, train_loader, val_loader, opt):
        
        model.to(self.device)
        model.train()
        self.logger.info('start training')
        for idx, data in enumerate(tqdm(train_loader)):
            break
            opt.zero_grad()
            video = data['video'].to(self.device)
            video = rearrange(video, 'n t c h w -> t n c h w') #batch first
            gloss = data['gloss'].to(self.device)
            video_length: torch.Tensor = data['video_length'].to(self.device)
            gloss_length: torch.Tensor = data['gloss_length'].to(self.device)
            y_predict = model(video, video_length)

            loss = self.loss_fn(y_predict, gloss, video_length, gloss_length)
            if self.debug and (np.isinf(loss.item()) or np.isnan(loss.item())):
                self.logger.debug('loss is nan or inf')
                self.logger.debug(f'annotation lenght {str(gloss_length)}')
                self.logger.debug(f'output size {str(y_predict.size())}')
                self.logger.debug(f'y_predict {str(y_predict)}')
                continue
            
            if self.device == 'cuda':
                self.scaler.scale(loss).backward()
                self.scaler.step(opt)
                self.scaler.update()
            else:
                loss.backward()
                opt.step()
                
            self.logger.info(f'iteration index: {idx}, batch loss: {loss.item()}')
        
        model.eval()
        ground_truth = []
        hypothesis = []
        for idx, data in enumerate(tqdm(val_loader)):
            video = data['video'].to(self.device)
            video = rearrange(video, 'n t c h w -> t n c h w') #batch first
            gloss = data['gloss'].to(self.device)
            video_length: torch.Tensor = data['video_length'].to(self.device)
            gloss_length: torch.Tensor = data['gloss_length'].to(self.device)
            y_predict = model(video, video_length)
            hypothesis = hypothesis + self.decoder(y_predict, video_length)
            ground_truth = ground_truth + self._get_ground_truth(gloss, gloss_length)
            
        wer_value = wer(ground_truth, hypothesis)
        self.logger.info(f'test finished, wer={wer_value}')
        
        return wer_value

            
        
        


            


