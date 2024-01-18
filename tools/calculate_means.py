import torch
import click
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
sys.path.append('src')
from csi_sign_language.data.dataset.phoenix14 import MyPhoenix14Dataset, CollateFn
from csi_sign_language.data.transforms.video import ToTensor

@click.command()
@click.option('--data_root', default='preprocessed/ph14_lmdb')
@click.option('--subset', default='multisigner')
@click.option('--device', default='cuda')
@click.option('--batch_size', default=2)
@click.option('--num_workers', default=6)
def main(data_root, subset, device, batch_size, num_workers):
    data_set = MyPhoenix14Dataset(data_root, subset, mode='train', transform=ToTensor())
    means = []
    vars = []
    n = len(data_set)
    loader = DataLoader(data_set, num_workers=num_workers, batch_size=batch_size, collate_fn=CollateFn())
    for data in tqdm(loader):
        video = data['video']
        video = video.to(device)
        #b t c h w
        means.append(torch.sum(torch.mean(video, dim=(1, -1, -2)), dim=0))
        vars.append(torch.sum(torch.var(video, dim=(1, -1, -2)), dim=0))

    means = torch.stack(means)
    vars = torch.stack(vars)
    mean = torch.sum(means, dim=0) / n
    vars = torch.sum(vars, dim=0) / n
    
    print(mean, vars.sqrt())
    
    
if __name__ == '__main__':
    main()