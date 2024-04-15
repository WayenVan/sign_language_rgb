from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from ..dataset.phoenix14 import MyPhoenix14Dataset
from ...data_utils.ph14.post_process import PostProcess
from ...data_utils.interface_post_process import IPostProcess
from ..dataset.phoenix14 import CollateFn


class Ph14DataModule(LightningDataModule):
    
    def __init__(self,
                 data_dir,
                 batch_size,
                 train_shuffle=True,
                 train_transform=None,
                 val_transform=None,
                 test_transform=None) -> None:
        super().__init__()
        self.data_root = data_dir
        self.batch_size = batch_size
        self.t_transform = train_transform
        self.v_transform = val_transform
        self.train_shuffle = train_shuffle

        if test_transform == None:
            test_transform = self.v_transform
    
    def setup(self, stage: str) -> None:
        self.train_set = MyPhoenix14Dataset(self.data_root, 'multisigner', 'train', transform=self.train_transform)
        self.val_set = MyPhoenix14Dataset(self.data_root, 'multisigner', 'dev', transform=self.train_transform)
        self.test_set = MyPhoenix14Dataset(self.data_root, 'multisigner', 'test', transform=self.train_transform)
    
    @property
    def post_process(self) -> IPostProcess:
        return PostProcess()

    def train_dataloader(self) -> Any:
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=self.train_shuffle, collate_fn=CollateFn())
        
    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, collate_fn=CollateFn())
    
    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, collate_fn=CollateFn())
    
    def predict_dataloader(self) -> TRAIN_DATALOADERS:
        return self.test_dataloader()