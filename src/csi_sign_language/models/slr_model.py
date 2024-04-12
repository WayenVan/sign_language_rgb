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
from csi_sign_language.data_utils.ph14.post_process import post_process
from csi_sign_language.modules.loss import VACLoss as _VACLoss
from csi_sign_language.modules.loss import HeatMapLoss

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
        self.save_hyperparameters(logger=False, ignore=['cfg', 'file_logger'])

        self.cfg = cfg
        self.data_excluded = getattr(cfg, 'data_excluded', [])
        self.backbone: nn.Module = instantiate(cfg.model)
        self.loss: nn.Module = instantiate(cfg.loss)

        self.vocab = vocab
        self.decoder = CTCDecoder(vocab, blank_id=0, search_mode=ctc_search_type, log_probs_input=True)
        
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
    def _extract_batch(batch):
        video = batch['video']
        gloss = batch['gloss']
        video_length = batch['video_length']
        gloss_length = batch['gloss_length']
        gloss_gt = batch['gloss_label']
        id = batch['id']
        return id, video, gloss, video_length, gloss_length, gloss_gt

    def forward(self, x, t_length) -> Any:
        return self.backbone(x, t_length)

    def training_step(self, batch, batch_idx):
        id, video, gloss, video_length, gloss_length, gloss_gt = self._extract_batch(batch)

        outputs = self.backbone(video, video_length)
        loss = self.loss(outputs, video, video_length, gloss, gloss_length)
        
        # if we should skip this batch
        skip_flag = torch.tensor(0, dtype=torch.uint8, device=self.device)
        if any(i in self.data_excluded for i in id):
            skip_flag = torch.tensor(1, dtype=torch.uint8, device=self.device)
        if torch.isnan(loss) or torch.isinf(loss):
            skip_flag = torch.tensor(1, dtype=torch.uint8, device=self.device)
        flags = self.all_gather(skip_flag)
        if any(f.item() for f in flags):
            del outputs
            del loss
            return None

        hyp = self._outputs2labels(outputs.out, outputs.t_length)
        hyp = self._gloss2sentence(hyp)
        gt  = self._gloss2sentence(gloss_gt)
        self.train_wer.update(hyp, gt)
        self.log('train_loss', loss, on_epoch=True, on_step=True)

        opt = self.optimizers(use_pl_optimizer=False)
        lr = opt.param_groups[0]['lr']
        self.log('lr', lr, on_step=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        id, video, gloss, video_length, gloss_length, gloss_gt = self._extract_batch(batch)

        with torch.inference_mode():
            outputs = self.backbone(video, video_length)
            loss = self.loss(outputs, video, video_length, gloss, gloss_length)

        hyp = self._outputs2labels(outputs.out, outputs.t_length)
        hyp = post_process(hyp, merge=True, regex=True)
        hyp = self._gloss2sentence(hyp)
        gt  = self._gloss2sentence(gloss_gt)
        self.val_wer.update(hyp, gt)
        self.log('val_loss', loss.detach(), on_epoch=False, on_step=True)
    
    def on_train_epoch_end(self):
        self.log('train_wer', self.train_wer.compute()*100)
        self.log('val_wer', self.val_wer.compute()*100)
        self.train_wer.reset()
        self.val_wer.reset()

    def configure_optimizers(self):
        opt: Optimizer = instantiate(self.cfg.optimizer, filter(lambda p: p.requires_grad, self.backbone.parameters()))
        scheduler = instantiate(self.cfg.lr_scheduler, opt)
        return {
            'optimizer': opt,
            'lr_scheduler': scheduler
        }
    
class VACLoss(nn.Module):

    def __init__(self, weights, temp, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss = _VACLoss(weights, temp)

    def forward(self, outputs, input, input_length, target, target_length): 
        conv_out = outputs.encoder_out.out
        conv_length = outputs.encoder_out.t_length
        seq_out = outputs.out
        t_length = outputs.t_length
        return self.loss(conv_out, conv_length, seq_out, t_length, target, target_length)

class MultiLoss(nn.Module):
    
    def __init__(self,
                 weights,
                 color_range,
                 cfg,
                 ckpt,
                 ) -> None:
        super().__init__()
        self.weights = weights
        self.pose_loss = HeatMapLoss(color_range, cfg, ckpt)
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='none')
    
    def forward(self, outputs, input, input_length, target, target_length): 
        #n c t h w
        heatmap_out = outputs.backbone_out.encoder_out.heatmap
        out = F.log_softmax(outputs.backbone_out.out, dim=-1)
        t_length = outputs.backbone_out.t_length
        loss = 0.
        
        if self.weights[0] > 0.:
            loss += self.ctc_loss(out, target.cpu().int(), t_length.cpu().int(), target_length.cpu().int()).mean()* self.weights[0]
        if self.weights[1] > 0.:
            heatmap_out = rearrange(heatmap_out, 'n c t h w -> (n t) c h w')
            input_ = rearrange(input, 'n c t h w -> (n t) c h w')
            loss += self.pose_loss(heatmap_out, input_) * self.weights[1]
        return loss 