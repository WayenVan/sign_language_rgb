import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import Module
import torchtext.vocab as v
from tqdm import tqdm
from einops import rearrange
import logging
from typing import *
from ..utils.data import *

class Inferencer():
    
    def __init__(
            self,
            vocab_dir,
            device,
            num_class
        ) -> None:
        
        self.vocab: v.Vocab = load_vocab(vocab_dir)
        self.device=device
        self.NUM_CLASS = num_class
    
    def do_inference(self, model, data_loader, parent_logger):
        hyp = []
        for data in tqdm(data_loader):
            hyp = hyp + self.do_inference_single_batch(model, data, parent_logger)
        return hyp
            
    def do_inference_single_batch(self, model, data, parent_logger: logging.Logger) -> List[List[str]]:
        logger = parent_logger.getChild(__class__.__name__) 
        
        model.to(self.device)
        model.eval()
        logger.info('start inference')
        
        video = data['video'].to(self.device)
        video = rearrange(video, 'n t c h w -> t n c h w') #batch first
        annotation = data['annotation'].to(self.device)
        video_mask: torch.Tensor = data['video_mask'].to(self.device)
        annotation_mask: torch.Tensor = data['annotation_mask'].to(self.device)
        y_predict = self.model(video, video_mask)
        hypothesis  = self.decoder(y_predict)
    
        return hypothesis
    
