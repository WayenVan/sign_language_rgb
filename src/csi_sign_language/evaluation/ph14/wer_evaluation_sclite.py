import os
import numpy as np
from typing import List



def get_phoenix_wer(work_dir, hyp, gt, tmp_prefix, res_dir):
    """calculate wer of ph14 dataset in a given working directory

    :param work_dir: the working directory
    :param hyp: the file name of hypothesis.ctm, it shoulde be relative directory to work_dir
    :param gt: the absulot directory of the groundtruth .stm file
    :param tmp_prefix: _description_
    :param res_dir: _description_
    :return: _description_
    """
    shell_file = os.path.join(res_dir, 'phoenix_eval.sh')
    python_file = os.path.join(res_dir, 'mergectmstm.py')
    
    cmd = "sh {:s} {:s} {:s} {:s} {:s} {:s}".format(shell_file, work_dir, hyp, gt, tmp_prefix, python_file)
    print(os.system(cmd))
    result_file = os.path.join(work_dir, '{:s}.out.{:s}.sys'.format(tmp_prefix, os.path.basename(hyp)))

    with open(result_file, 'r') as fid:
        for line in fid:
            line = line.strip()
            if 'Sum/Avg' in line:
                result = line
                break
    tmp_err = result.split('|')[3].split()
    subs, inse, dele, wer = tmp_err[1], tmp_err[3], tmp_err[2], tmp_err[4]
    subs, inse, dele, wer = float(subs), float(inse), float(dele), float(wer)
    errs = [wer, subs, inse, dele]
    # os.system('rm {:s}'.format(os.path.join(shell_dir, '{:s}.tmp.*'.format(tmp_prefix))))
    # os.system('rm {:s}'.format(os.path.join(shell_dir, hyp)))
    return errs

def glosses2ctm(ids: List[str], glosses: List[List[str]], path: str):
    with open(path, 'w') as f:
        for id, gloss in list(zip(ids, glosses)):
            start_time = 0
            for single_gloss in gloss:
                tl = np.random.random() * 0.1
                f.write(
                    '{} 1 {:.3f} {:.3f} {}\n'.format(id, start_time, start_time + tl, single_gloss)
                )
                start_time += tl

def eval(ids, work_dir, hypothesis, stm_path, ctm_file_name, res_dir):
    glosses2ctm(ids, hypothesis, os.path.join(work_dir, ctm_file_name))
    re = get_phoenix_wer(work_dir,ctm_file_name, stm_path, tmp_prefix='.', res_dir=res_dir)