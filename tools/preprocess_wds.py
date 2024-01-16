import click
import os
import pandas as pd
from torchtext.vocab import build_vocab_from_iterator
import sys
sys.path.append('src')

from tqdm import tqdm
import glob
from typing import *
import numpy as np
import cv2
from einops import rearrange
from csi_sign_language.utils.data import VideoGenerator
from omegaconf import OmegaConf
import json
import webdataset as wds
import numpy as np


@click.command()
@click.option('--data_root', default='dataset/phoenix2014')
@click.option('--output_root', default='preprocessed/ph14')
@click.option('--frame_size', nargs=2, default=(224, 224))
@click.option('--subset', default='multisigner')
def main(data_root, output_root, frame_size, subset):
    
    vocab, vocab_SI5 = generate_vocab(data_root, output_root)
    info = OmegaConf.create()
    info.author = 'jingyan wang'
    info.email = '2533494w@student.gla.ac.uk'
    
    subset_root = os.path.join(output_root, subset)
    multi_signer = True if subset == 'multisigner' else False
    v = vocab if subset == 'multisigner' else vocab_SI5
    info['vocab'] = v.get_itos()
    os.makedirs(subset_root, exist_ok=True)
    for mode in ('train', 'dev', 'test'):
        sink = wds.TarWriter(
            os.path.join('preprocessed/ph14', subset, f'{mode}.tar')
        )
        annotations, feature_dir, max_lgt_v, max_lgt_g= get_basic_info(data_root, mode, multisigner=multi_signer)
        data_length = len(annotations)
        print(f'creating {subset}-{mode}')
        info[mode] = {}
        info[mode]['max_length_video'] = max_lgt_v
        info[mode]['max_length_gloss'] = max_lgt_g
        for idx in tqdm(range(data_length)):
            video, gloss = get_single_data(idx, annotations, v, feature_dir, frame_size)
            sink.write({
                '__key__': str(idx),
                'video': video.tobytes(),
                'gloss': gloss.tobytes(),
                'video_shape': np.array(video.shape).tobytes(),
                'gloss_shape': np.array(gloss.shape).tobytes(),
                'video_dtype': str(video.dtype),
                'gloss_dtype': str(gloss.dtype)
            })

    info_d = OmegaConf.to_container(info)
    with open(os.path.join(subset_root, 'info.json'), 'w') as f:
        json.dump(info_d, f)

            

    
def get_single_data(idx, annotations, gloss_vocab,feature_dir, frame_size=(224, 224)):
    anno = annotations['annotation'].iloc[idx]
    anno: List[str] = anno.split()
    anno: List[int] = gloss_vocab(anno)
    anno: np.ndarray = np.asarray(anno)

    folder: str = annotations['folder'].iloc[idx]
    frame_files: List[str] = get_frame_file_list_from_annotation(feature_dir, folder)

    video_gen: VideoGenerator = VideoGenerator(frame_files)
    frames: List[np.ndarray] = [cv2.resize(frame, frame_size) for frame in video_gen]
    # [t, h, w, c]
    frames: np.ndarray = np.stack(frames)

    # padding
    # frames, frames_mask = padding(frames, 0, length_video, 'back')
    # anno, anno_mask = padding(anno, 0, length_gloss, 'back')
    
    frames = rearrange(frames, 't h w c -> t c h w')
    
    
    return frames, anno
    
    
def get_basic_info(data_root, type='train', multisigner=True,):
    
    if multisigner:
        annotation_dir = os.path.join(data_root, 'phoenix-2014-multisigner/annotations/manual')
        annotation_file = type + '.corpus.csv'
        feature_dir = os.path.join('phoenix-2014-multisigner/features/fullFrame-210x260px', type)
    else:
        annotation_dir = os.path.join(data_root, 'phoenix-2014-signerindependent-SI5/annotations/manual')
        annotation_file = type + '.SI5.corpus.csv'
        feature_dir = os.path.join('phoenix-2014-signerindependent-SI5/features/fullFrame-210x260px', type)

    annotations = pd.read_csv(os.path.join(annotation_dir, annotation_file), delimiter='|')
    feature_dir = os.path.join(data_root, feature_dir)

    max_lgt_vid = max_length_time(annotations, feature_dir)
    max_lgt_gloss = max_length_gloss(annotations)
    
    return annotations, feature_dir, max_lgt_vid, max_lgt_gloss

def get_frame_file_list_from_annotation(feature_dir, folder: str) -> List[str]:
    """return frame file list with the frame order"""
    file_list: List[str] = glob.glob(os.path.join(feature_dir, folder))
    file_list = sorted(file_list, key=lambda x: int(x.split('_')[-1].split('-')[0][2:]))
    return file_list


def max_length_time(annotations, feature_dir):
    max = 0
    for folder in annotations['folder']:
        file_list = glob.glob(os.path.join(feature_dir, folder))
        if len(file_list) >= max:
            max = len(file_list)
    return max

def max_length_gloss(annotations):
    max = 0
    for glosses in annotations['annotation']:
        l = len(glosses.split())
        if l > max:
            max = l
    return max

def create_glossdictionary(annotations):
    def tokens():
        for annotation in annotations['annotation']:
            yield annotation.split()
    vocab = build_vocab_from_iterator(tokens(), special_first=True, specials=['<PAD>'])
    return vocab
    
def generate_vocab(data_root, output_root):
    print(os.getcwd())
    annotation_file_multi = os.path.join(data_root, 'phoenix-2014-multisigner/annotations/manual')
    annotation_file_single = os.path.join(data_root, 'phoenix-2014-signerindependent-SI5/annotations/manual')
    
    multi = []
    si5 = []
    for type in ('dev', 'train', 'test'):
        multi.append(pd.read_csv(os.path.join(annotation_file_multi, type+'.corpus.csv'), delimiter='|'))
        si5.append(pd.read_csv(os.path.join(annotation_file_single, type+'.SI5.corpus.csv'), delimiter='|'))
    
    vocab_multi = create_glossdictionary(pd.concat(multi))
    vocab_single =  create_glossdictionary(pd.concat(si5))
    
    return vocab_multi, vocab_single


if __name__ == '__main__':
    main()