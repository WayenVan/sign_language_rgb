import click
from functools import partial
import os
import pandas as pd
from torchtext.vocab import build_vocab_from_iterator
import sys
from multiprocessing import Pool
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
from csi_sign_language.utils.lmdb_tool import store_numpy_array, retrieve_numpy_array
import lmdb



@click.command()
@click.option('--data_root', default='dataset/phoenix2014-release')
@click.option('--output_root', default='preprocessed/ph14')
@click.option('--frame_size', nargs=2, default=(256, 256))
@click.option('--subset', default='multisigner')
@click.option('--multiprocess', default=True)
@click.option('--chunk_size', default=4, help='the chunk size of data submitted to each process to handle')
def main(data_root, output_root, frame_size, subset, multiprocess, chunk_size):
    
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
        annotations, feature_dir, max_lgt_v, max_lgt_g= get_basic_info(data_root, mode, multisigner=multi_signer)
        data_length = len(annotations)
        print(f'creating {subset}-{mode}')
        info[mode] = {}
        info[mode]['max_length_video'] = max_lgt_v
        info[mode]['max_length_gloss'] = max_lgt_g
        if multiprocess:
            keys = run_mp_cmd(partial(process_data, mode, v, frame_size, annotations, feature_dir, subset, subset_root), range(data_length), chunk_size=chunk_size)
        else:
            keys = process_data(list(range(data_length)), info, mode, v, frame_size, annotations, feature_dir, subset, subset_root)
        
        info[mode]['data'] = keys

    info_d = OmegaConf.to_container(info)
    with open(os.path.join(subset_root, 'info.json'), 'w') as f:
        json.dump(info_d, f, indent=4)
            
def process_data(mode, v, frame_size, annotations, feature_dir, subset, subset_root, idxes):
    env = lmdb.open(os.path.join(subset_root, 'feature_database'), map_size=1099511627776)
    keys = []
    for idx in idxes:
        video, gloss, gloss_label = get_single_data(idx, annotations, v, feature_dir, frame_size)
        video_key = f'{subset}-{mode}-{idx}-video'
        gloss_key = f'{subset}-{mode}-{idx}-gloss'
        keys.append(dict(
            video_key=video_key,
            gloss_key=gloss_key,
            video_shape=video.shape,
            video_dtype=str(video.dtype),
            gloss_shape=gloss.shape,
            gloss_dtype=str(gloss.dtype),
            gloss_label=gloss_label
        ))
        store_numpy_array(env, video_key, video)
        store_numpy_array(env, gloss_key, gloss)
    env.close()
    return keys

def run_mp_cmd(func, data_indexes, chunk_size):
    process_args = [data_indexes[i:i + chunk_size] for i in range(0, len(data_indexes), chunk_size)]  
    with Pool() as p:
        keys = []
        for result in tqdm(p.imap(func, process_args), total=len(process_args)):
            keys += result 
    return keys


def get_single_data(idx, annotations, gloss_vocab,feature_dir, frame_size=(224, 224)):
    anno = annotations['annotation'].iloc[idx]
    anno_str: List[str] = anno.split()
    anno: List[int] = gloss_vocab(anno_str)
    anno: np.ndarray = np.asarray(anno)

    folder: str = annotations['folder'].iloc[idx]
    frame_files: List[str] = get_frame_file_list_from_annotation(feature_dir, folder)

    video_gen: VideoGenerator = VideoGenerator(frame_files)
    frames: List[np.ndarray] = [cv2.cvtColor(cv2.resize(frame, frame_size, cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB) for frame in video_gen]
    # [t, h, w, c]
    frames: np.ndarray = np.stack(frames)

    # padding
    # frames, frames_mask = padding(frames, 0, length_video, 'back')
    # anno, anno_mask = padding(anno, 0, length_gloss, 'back')
    
    frames = rearrange(frames, 't h w c -> t c h w')
    
    
    return frames, anno, anno_str
    
    
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
        data_indexes = len(glosses.split())
        if data_indexes > max:
            max = data_indexes
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