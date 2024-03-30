
from omegaconf import OmegaConf
from hydra.utils import instantiate
import sys
import os
os.chdir('/home/jingyan/Documents/sign_language_rgb')
sys.path.append('src')


from csi_sign_language.models.slr_ctc_baseline import SLRModel