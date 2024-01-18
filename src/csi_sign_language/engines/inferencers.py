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
            vocab,
            device,
            num_class,
            logger: logging.Logger
        ) -> None:
        
        self.vocab: v.Vocab = vocab
        self.device=device
        self.NUM_CLASS = num_class
        self.logger = logger.getChild(__class__.__name__)
        self.decoder = CTCDecoder(self.vocab, blank_id=0, search_mode='greedy')
    def do_inference(self, model: Module, loader) -> List[List[str]]:
        
        model.eval()
        
        ground_truth = []
        hypothesis = []
        for idx, data in enumerate(tqdm(loader)):
            video = data['video'].to(self.device)
            video = rearrange(video, 'n t c h w -> t n c h w') #batch first
            gloss = data['gloss'].to(self.device)
            video_length: torch.Tensor = data['video_length'].to(self.device)
            gloss_length: torch.Tensor = data['gloss_length'].to(self.device)
            with torch.no_grad():
                outputs = model(video, video_length)
                y_predict = outputs['seq_out']
                video_length = outputs['video_length']

            hypothesis += self.decoder(y_predict, video_length)
            ground_truth += self._get_ground_truth(gloss, gloss_length)
        return hypothesis, ground_truth
    

    def _get_ground_truth(self, ground_truth, gloss_length) -> List[List[str]]:
        """

        :param ground_truth: [n t]
        :param sequence_masks: [n]
        """
        ret = []
        for batch_id, gloss in enumerate(ground_truth):
            gloss = gloss.cpu().numpy()
            l = gloss_length[batch_id].cpu().numpy()
            temp = self.vocab.lookup_tokens(gloss[:l])
            if len(temp) == 0:
                a = 1
            ret.append(temp)
        return ret