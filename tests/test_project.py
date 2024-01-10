import sys
sys.path.append('src')
from omegaconf import OmegaConf
from csi_sign_language.data.build import build_dataloader
from csi_sign_language.utils.metrics import leven_dist
from tqdm import tqdm

def test_graphsetgment_dataset():
    """testing if all the dataloader is correct given a default config file
    """
    cfg = OmegaConf.load('configs/default.yaml') 
    loader = build_dataloader(cfg)
    for data in tqdm(loader['train_loader']):
        pass
    
    for data in tqdm(loader['test_loader']):
        pass

    for data in tqdm(loader['val_loader']):
        pass


def test_wer():
    assert leven_dist('kitten', 'KiTtEn') == 3
    assert leven_dist('', '') == 0
    assert leven_dist('sunny day', 'rainy day') == 3