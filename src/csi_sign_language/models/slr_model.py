import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Any
from einops import rearrange
from ..utils.decode import CTCDecoder
from ..modules.slr_base.base_stream import BaseStream
from collections import namedtuple
from hydra.utils import instantiate

import lightning as L
from omegaconf.dictconfig import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from csi_sign_language.evaluation.ph14.post_process import post_process

from torchmetrics import WordErrorRate
from typing import List

class SLRModel(L.LightningModule):

    def __init__(self, 
                 cfg: DictConfig,
                 vocab,
                 ctc_search_type = 'greedy',
                 file_logger = None,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(logger=False, ignore=['file_logger'])

        self.cfg = cfg
        self.data_excluded = getattr(cfg, 'data_excluded', [])
        self.backbone: nn.Module = instantiate(cfg.model)
        self.loss: nn.Module = instantiate(cfg.loss)

        self.decoder = CTCDecoder(self.vocab, blank_id=0, search_mode=ctc_search_type, log_probs_input=True)
        self.vocab = vocab
        
        self.train_wer = WordErrorRate(sync_on_compute=True)
        self.val_wer = WordErrorRate(sync_on_compute=True)
        
    @torch.no_grad()
    def _outputs2labels(self, out, length):
        #[t n c]
        #return list(list(string))
        y_predict = torch.nn.functional.log_softmax(out, -1).detach().cpu()
        return self.decoder(y_predict, length)

    @staticmethod
    def _gloss2sentence(gloss: List[List[str]]):
        return [' '.join(g) for g in gloss]
    
    @staticmethod
    def _extract_batch(self, batch):
        video = batch['video']
        gloss = batch['gloss']
        video_length = batch['video_length']
        gloss_length = batch['gloss_length']
        gloss_gt = batch['gloss']
        id = batch['id']
        return id, video, gloss, video_length, gloss_length, gloss_gt

    def forward(self, x, t_length) -> Any:
        return self.backbone(x, t_length)

    def training_step(self, batch, batch_idx):
        id, video, gloss, video_length, gloss_length, gloss_gt = self._extract_batch(batch)

        outputs = self.backbone(video, video_length)
        loss = self.loss(outputs, video, video_length, gloss, gloss_length)
        
        # if we should skip this batch
        skip_flag = torch.tensor(0, dtype=torch.uint8)
        if any(i in self.data_excluded for i in id):
            skip_flag = torch.tensor(1, dtype=torch.uint8)
        if torch.isnan(loss) or torch.isinf(loss):
            skip_flag = torch.tensor(1, dtype=torch.uint8)
        flags = self.all_gather(skip_flag)
        if any(f.item() for f in flags):
            del outputs
            del loss
            return None

        hyp = self._outputs2labels(outputs.out, outputs.t_length)
        hyp = self._gloss2sentence(hyp)
        self.train_wer.update(hyp, gloss_gt)
        self.log('train_loss', loss, on_epoch=True, on_step=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        id, video, gloss, video_length, gloss_length, gloss_gt = self._extract_batch(batch)

        with torch.inference_mode():
            outputs = self.backbone(video, video_length)
            loss = self.loss(outputs, video, video_length, gloss, gloss_length)

        hyp = self._outputs2labels(outputs.out, outputs.t_length)
        hyp = post_process(hyp, merge=True, regex=True)
        hyp = self._gloss2sentence(hyp)
        self.val_wer.update(hyp, gloss_gt)
        self.log('val_loss', loss, on_epoch=True, on_step=True)
    
    def on_train_epoch_end(self):
        self.train_wer.reset()
        self.val_wer.reset()

    def configure_optimizers(self):
        opt: Optimizer = instantiate(self.cfg.optimizer, filter(lambda p: p.requires_grad, self.backbone.parameters()))
        scheduler = instantiate(self.cfg.lr_scheduler, opt)
        return {
            'optimizer': opt,
            'lr_scheduler': scheduler
        }
    
    