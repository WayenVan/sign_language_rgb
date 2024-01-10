from .dataset.phoenix14 import *
from torch.utils.data import Subset, DataLoader
from omegaconf import OmegaConf, DictConfig
from .transforms.transforms import ImageResize
import os
import torchvision.transforms as T

def build_dataset(cfg: DictConfig):

    data_root = cfg.data.phoenix14_root
    img_transform = T.Compose([
        T.ToPILImage(),
        T.Resize(size=(cfg.data.frame_size.h, cfg.data.frame_size.w)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_set = Phoenix14Dataset(
        data_root=data_root,
        gloss_vocab_dir=cfg.data.vocab_dir,
        type='train',
        img_transform=img_transform
    )

    val_set = Phoenix14Dataset(
        data_root=data_root,
        gloss_vocab_dir=cfg.data.vocab_dir,
        type='dev',
        img_transform=img_transform
    )
    test_set = Phoenix14Dataset(
        data_root=data_root,
        gloss_vocab_dir=cfg.data.vocab_dir,
        type='test',
        img_transform=img_transform
    )

    return dict(
        train_set = train_set,
        val_set = val_set,
        test_set = test_set
    )

def build_dataloader(cfg: DictConfig):
    dataset = build_dataset(cfg)
    b_train = cfg.data.train_loader.batch_size
    b_val = cfg.data.validate_loader.batch_size
    b_test = cfg.data.test_loader.batch_size
    return dict(
        train_loader = DataLoader(dataset['train_set'], batch_size=b_train, num_workers=cfg.data.num_workers),
        val_loader = DataLoader(dataset['val_set'], batch_size=b_val, num_workers=cfg.data.num_workers),
        test_loader = DataLoader(dataset['test_set'], batch_size=b_test, num_workers=cfg.data.num_workers)
    )

