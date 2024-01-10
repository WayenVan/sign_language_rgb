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
        vocab_dir: str,
        save_directory: str,
        device,
        ) -> None:

        self.NUM_CLASS = int(num_class)
        self.device = device
        self.save_directory = save_directory
        self.loss_fn = nn.CTCLoss(zero_infinity=True)
        self.vocab = load_vocab(vocab_dir)
        self.decoder = CTCDecoder(self.vocab, blank_id=0)
    
    def _get_ground_truth(self, ground_truth, sequence_masks) -> List[List[str]]:
        ret = []
        for batch_id, output in enumerate(ground_truth):
            output = output.cpu().numpy()
            sequence_mask = sequence_masks[batch_id].cpu().numpy()
            ret.append(self.vocab.lookup_tokens(output[sequence_mask]))
        return ret
        
    def do_train(self, model, train_loader, val_loader, opt, parent_logger):
        logger = parent_logger.getChild(__class__.__name__) 
        
        model.to(self.device)
        model.train()
        logger.info('start training')
        # for idx, data in enumerate(tqdm(train_loader)):
        #     opt.zero_grad()
        #     video = data['video'].to(self.device)
        #     video = rearrange(video, 'n t c h w -> t n c h w') #batch first
        #     annotation = data['annotation'].to(self.device)
        #     video_mask: torch.Tensor = data['video_mask'].to(self.device)
        #     annotation_mask: torch.Tensor = data['annotation_mask'].to(self.device)
        #     y_predict = model(video, video_mask)
        #     loss = self.loss_fn(y_predict, annotation, video_mask.sum(-1), annotation_mask.sum(-1))
        #     loss.backward()
        #     opt.step()
        #     logger.info(f'iteration index: {idx}, batch loss: {loss.item()}')
        
        model.eval()
        ground_truth = []
        hypothesis = []
        for idx, data in enumerate(tqdm(val_loader)):
            video = data['video'].to(self.device)
            video = rearrange(video, 'n t c h w -> t n c h w') #batch first
            annotation = data['annotation'].to(self.device)
            video_mask: torch.Tensor = data['video_mask'].to(self.device)
            annotation_mask: torch.Tensor = data['annotation_mask'].to(self.device)
            y_predict = model(video, video_mask)
            hypothesis = hypothesis + self.decoder(y_predict, video_mask.sum(dim=-1))
            ground_truth = ground_truth + self._get_ground_truth(annotation, annotation_mask)
            
        wer_value = wer(ground_truth, hypothesis)
        logger.info(f'test finished, wer={wer_value}')
        
        return wer_value

            
        
        


            


