import torch
from torch.nn import Module
from tqdm import tqdm
import logging
from typing import *
from ..utils.data import *
from ..utils.misc import *
class Inferencer():
    
    def __init__(
            self,
            device,
            logger: logging.Logger,
            use_amp=True
        ) -> None:
        assert isinstance(device, str), 'device must be a string'

        self.device=device
        self.use_amp = use_amp
        if 'cuda' in self.device:
            self.is_cuda = True
        else:
            self.is_cuda = False

        if logger is not None:
            self.logger: logging.Logger = logger.getChild(__class__.__name__)
        else:
            self.logger = None

    def do_inference(self, model: Module, loader) -> List[List[str]]:

        model.to(self.device)
        model.eval()
        ids = []
        ground_truth = []
        hypothesis = []

        for idx, data in enumerate(tqdm(loader)):

            video = data['video'].to(self.device)
            video_length: torch.Tensor = data['video_length'].to(self.device)

            with torch.inference_mode():
                if self.use_amp and self.device == 'cuda':
                    with torch.autocast('cuda'):
                        hyp = model.inference(video, video_length)
                else:
                    hyp = model.inference(video, video_length)

            hypothesis += hyp
            ground_truth += data['gloss_label']
            ids += data['id']
            
        return ids, hypothesis, ground_truth
    