import torch
import ctcdecode
import sys

sys.path.append('src')

from csi_sign_language.data.build import *
from csi_sign_language.models.build_model import *
from csi_sign_language.utils.inspect import *
from csi_sign_language.utils.logger import build_logger
from csi_sign_language.utils.metrics import leven_dist
from csi_sign_language.utils.decode import CTCDecoder

from torchinfo import summary

cfg = OmegaConf.load('configs/default.yaml') 
loader = build_dataloader(cfg)['train_loader']
logger = build_logger('main', 'experiments/log.log')
model =  build_model(cfg)
data = next(iter(loader))
data['video'] = rearrange(data['video'], 'n t c h w -> t n c h w')
output = model(data['video'])
voc = loader.dataset.gloss_vocab
decode = CTCDecoder(
    vocab=voc,
    search_mode='greedy'
)

result = decode(output, torch.sum(data['video_mask'], dim=-1))
a = 1