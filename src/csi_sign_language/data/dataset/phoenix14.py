import os
import numpy as np
import glob
import cv2 as cv2
import pandas as pd
from typing import Any, Tuple, List
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchtext.vocab import vocab, build_vocab_from_iterator, Vocab

from einops import rearrange

from csi_sign_language.csi_typing import PaddingMode
from ...csi_typing import *
from ...utils.data import VideoGenerator, padding, load_vocab

from abc import ABC, abstractmethod


class BasePhoenix14Dataset(Dataset, ABC):
    data_root: str
    def __init__(self, data_root, gloss_vocab_dir, type='train', multisigner=True, length_time=None, length_glosses=None,
                padding_mode : PaddingMode ='front'):
        if multisigner:
            annotation_dir = os.path.join(data_root, 'phoenix-2014-multisigner/annotations/manual')
            annotation_file = type + '.corpus.csv'
            feature_dir = os.path.join('phoenix-2014-multisigner/features/fullFrame-210x260px', type)
        else:
            annotation_dir = os.path.join(data_root, 'phoenix-2014-signerindependent-SI5/annotations/manual')
            annotation_file = type + '.SI5.corpus.csv'
            feature_dir = os.path.join('phoenix-2014-signerindependent-SI5/features/fullFrame-210x260px', type)

        self._annotations = pd.read_csv(os.path.join(annotation_dir, annotation_file), delimiter='|')
        self._feature_dir = feature_dir
        self._data_root = data_root

        self._length_time = self.max_length_time if length_time == None else length_time
        self._length_gloss = self.max_length_gloss if length_glosses == None else length_glosses
        self._padding_mode = padding_mode

        self.gloss_vocab = load_vocab(gloss_vocab_dir)

    def __len__(self):
        return len(self._annotations)

    @abstractmethod
    def __getitem__(self, idx):
        return

    @property
    def max_length_time(self):
        max = 0
        for folder in self._annotations['folder']:
            file_list = glob.glob(os.path.join(self._data_root, self._feature_dir, folder))
            if len(file_list) >= max:
                max = len(file_list)
        return max

    @property
    def max_length_gloss(self):
        max = 0
        for glosses in self._annotations['annotation']:
            l = len(glosses.split())
            if l > max:
                max = l
        return max
    

class Phoenix14Dataset(BasePhoenix14Dataset):
    """
    Dataset for general RGB image with gloss label, the output is (frames, gloss_labels, frames_padding_mask, gloss_padding_mask)
    """
    def __init__(self, data_root, gloss_vocab_dir, type='train', multisigner=True, length_time=None, length_glosses=None,img_transform=None, transform=None):
        super().__init__(data_root, gloss_vocab_dir, type, multisigner, length_time, length_glosses, 'back')
        self._img_transform = img_transform
        self.transform = transform

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        anno = self._annotations['annotation'].iloc[idx]
        anno: List[str] = anno.split()
        anno: List[int] = self.gloss_vocab(anno)
        anno: np.ndarray = np.asarray(anno)

        folder: str = self._annotations['folder'].iloc[idx]
        frame_files: List[str] = self._get_frame_file_list_from_annotation(folder)

        video_gen: VideoGenerator = VideoGenerator(frame_files)
        frames: List[np.ndarray] = [frame if self._img_transform == None else self._img_transform(frame)  for frame in video_gen]
        # [t, h, w, c]
        frames: np.ndarray = np.stack(frames)

        # padding
        frames, frames_mask = padding(frames, 0, self._length_time, self._padding_mode)
        anno, anno_mask = padding(anno, 0, self._length_gloss, self._padding_mode)
        
        
        ret = dict(
            video=frames, #[t, h, w, c]
            annotation = anno, #[s]
            video_mask = frames_mask, #[t]
            annotation_mask = anno_mask #[s]
        )

        if self.transform:
            ret = self.transform(ret)
            
        return ret
    
    def _get_frame_file_list_from_annotation(self, folder: str) -> List[str]:
        """return frame file list with the frame order"""
        file_list: List[str] = glob.glob(os.path.join(self._data_root, self._feature_dir, folder))
        file_list = sorted(file_list, key=lambda x: int(x.split('_')[-1].split('-')[0][2:]))
        return file_list
        
