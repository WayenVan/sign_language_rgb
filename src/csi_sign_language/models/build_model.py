from .models import *
from omegaconf import OmegaConf, DictConfig

def build_model(cfg: DictConfig):
    model = ResnetTransformer(
        d_feedforward=cfg.model.dim_feed,
        n_head=cfg.model.n_head,
        n_class=cfg.model.n_class
    )
    return model