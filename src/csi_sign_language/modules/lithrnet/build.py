import torch
from .litehrnet import LiteHRNet
def build_litehrnet(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = LiteHRNet(**checkpoint['initial_args'])
    model.load_state_dict(checkpoint['state_dict'])
    return model
    
    