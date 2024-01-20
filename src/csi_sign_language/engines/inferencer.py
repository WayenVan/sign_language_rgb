import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import Module
import torchtext.vocab as v
from tqdm import tqdm
from einops import rearrange
import logging
from typing import *
from ..utils.data import *
from csi_sign_language.utils.decode import CTCDecoder
class Inferencer():
    
    def __init__(
            self,
            device,
            num_class,
            logger: logging.Logger
        ) -> None:
        
        self.device=device
        self.NUM_CLASS = num_class
        self.logger = logger.getChild(__class__.__name__)

    def do_inference(self, model: Module, loader) -> List[List[str]]:
        
        model.eval()
        ground_truth = []
        hypothesis = []
        for idx, data in enumerate(tqdm(loader)):
            video = data['video'].to(self.device)
            video = rearrange(video, 'n t c h w -> t n c h w') #batch first
            video_length: torch.Tensor = data['video_length'].to(self.device)
            with torch.no_grad():
                hyp = model.inference(video, video_length)
            hypothesis += hyp
            ground_truth += data['gloss_label']
            
        return hypothesis, ground_truth
    
