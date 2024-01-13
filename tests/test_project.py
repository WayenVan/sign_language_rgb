import sys
sys.path.append('src')
from omegaconf import OmegaConf
from csi_sign_language.utils.metrics import leven_dist
from tqdm import tqdm

def test_wer():
    assert leven_dist('kitten', 'KiTtEn') == 3
    assert leven_dist('', '') == 0
    assert leven_dist('sunny day', 'rainy day') == 3