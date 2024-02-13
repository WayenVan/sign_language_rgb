from omegaconf import OmegaConf, DictConfig
import sys
import logging
sys.path.append('src')
from hydra.utils import instantiate
from matplotlib import pyplot as plt
import torch
from csi_sign_language.engines.inferencer import Inferencer
from csi_sign_language.evaluation.ph14.post_process import post_process
from csi_sign_language.evaluation.ph14.wer_evaluation_sclite import glosses2ctm, get_phoenix_wer, eval
from csi_sign_language.evaluation.ph14.wer_evaluation_python import wer_calculation
import hydra
import os
import json
logger = logging.getLogger('main')

@hydra.main(version_base=None, config_path='../configs/evaluate', config_name='default.yaml')
def main(cfg: DictConfig):
    result = OmegaConf.create()
    work_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    train_cfg = OmegaConf.load(cfg.train_config)

    test_loader = instantiate(cfg.data.test_loader)
    vocab = test_loader.dataset.get_vocab()
    
    model: torch.nn.Module = instantiate(train_cfg.model, vocab=vocab)
    model.to(cfg.device)
    checkpoint = torch.load(cfg.checkpoint)
    model.load_state_dict(checkpoint['model_state'])
    
    inferencer: Inferencer = instantiate(cfg.inferencer, logger=logger)
    ids, hypothesis, ground_truth = inferencer.do_inference(model, test_loader)
    
    ret = []
    for id, gt, pre, post in list(zip(ids, ground_truth, hypothesis, post_process(hypothesis))):
        ret.append(dict(
            id=id,
            gt=gt,
            pre=pre,
            post=post
        ))
    with open(os.path.join(work_dir, 'result.json'), 'w') as f:
        json.dump(ret, f, indent=4)

    print(wer_calculation(ground_truth, post_process(hypothesis)))
    
    #better detail provided by sclite. need to merge for better performance
    wer_value = eval(ids, work_dir, post_process(hypothesis, regex=False, merge=True), test_loader.dataset.get_stm(), 'hyp.ctm', cfg.evaluation_tool)
    print(wer_value[0])
    
    meta = checkpoint['meta']
    epoch = [m['epoch'] for m in meta]
    loss = [m['train_loss'] for m in meta]
    wer = [m['val_wer'] for m in meta]
    fig, axe = plt.subplots()
    lins1 = axe.plot(epoch, loss, label='loss')
    axe2 = axe.twinx()
    lins2 = axe2.plot(epoch, wer, color='r', label='WER')
    
    lns = lins1 + lins2
    labs = [l.get_label() for l in lns]
    axe.legend(lns, labs, loc=0)

    axe.set_xlabel('epoch')
    axe.set_ylabel('loss')
    axe2.set_ylabel('WER')
    fig.savefig(os.path.join(work_dir, 'fig.png'))
    
    


if __name__ == '__main__':
    main()