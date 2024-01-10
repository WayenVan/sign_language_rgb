import click
import os
import pandas as pd
import json
from torchtext.vocab import build_vocab_from_iterator
import sys

@click.command()
@click.option('--data_root', default='dataset/phoenix2014-release')
@click.option('--output_root', default='dataset')
def main(data_root, output_root):
    print(os.getcwd())
    annotation_file_multi = os.path.join(data_root, 'phoenix-2014-multisigner/annotations/manual')
    annotation_file_single = os.path.join(data_root, 'phoenix-2014-signerindependent-SI5/annotations/manual')
    
    multi = []
    si5 = []
    for type in ('dev', 'train', 'test'):
        multi.append(pd.read_csv(os.path.join(annotation_file_multi, type+'.corpus.csv'), delimiter='|'))
        si5.append(pd.read_csv(os.path.join(annotation_file_single, type+'.SI5.corpus.csv'), delimiter='|'))
    
    vocab_multi = _create_glossdictionary(pd.concat(multi))
    vocab_single =  _create_glossdictionary(pd.concat(si5))
    
    with open(os.path.join(output_root, 'pheonix14-multi-vocab.txt'), 'w') as f:
        f.writelines(item+'\n' for item in vocab_multi.get_itos())
    
    with open(os.path.join(output_root, 'pheonix14-SI5-vocab.txt'), 'w') as f:
        f.writelines(item+'\n' for item in vocab_single.get_itos())
    

def _create_glossdictionary(annotations):
    def tokens():
        for annotation in annotations['annotation']:
            yield annotation.split()
    vocab = build_vocab_from_iterator(tokens(), special_first=True, specials=['<PAD>'])
    return vocab

if __name__ == '__main__':
    main()