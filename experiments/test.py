
from omegaconf import OmegaConf
from hydra.utils import instantiate
import sys
import os
os.chdir('/home/jingyan/Documents/sign_language_rgb')
sys.path.append('src')

from csi_sign_language.data.transforms.common import TemporalAug
config = OmegaConf.load('configs/train/x3d_trans.yaml')
dataset = instantiate(config.data.train_set)

dataset[0]['video'].dtype
from torchvision.transforms import InterpolationMode
print(InterpolationMode.NEAREST)
from torchvision.transforms.functional import adjust_brightness