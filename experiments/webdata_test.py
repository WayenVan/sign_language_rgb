import omegaconf as om
import hydra
import sys
sys.path.append('src')
cfg = om.OmegaConf.load('/home/wayenvan/Documents/sign_language_rgb/configs/train/default_wds.yaml')

a = hydra.utils.instantiate(cfg.data.val_set)

for item in a:
    print(item)