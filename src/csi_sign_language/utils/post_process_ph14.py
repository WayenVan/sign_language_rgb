from typing import List
from itertools import groupby
import re

def post_process(output: List[List[str]]):

    return [apply_regex(item) for item in output]

def apply_regex(output: List[str]):
    output_s = ' '.join(output)

    output_s = re.sub(r'loc-', r'', output_s)
    output_s = re.sub(r'cl-', r'', output_s)
    output_s = re.sub(r'qu-', r'', output_s)
    output_s = re.sub(r'poss-', r'', output_s)
    output_s = re.sub(r'lh-', r'', output_s)
    output_s = re.sub(r'S0NNE', r'SONNE', output_s)
    output_s = re.sub(r'HABEN2', r'HABEN', output_s)

    output_s = re.sub(r'__EMOTION__', r'', output_s)
    output_s = re.sub(r'__PU__', r'', output_s)
    output_s = re.sub(r'__LEFTHAND__', r'', output_s)
    
    output_s = re.sub(r'WIE AUSSEHEN', r'WIE-AUSSEHEN', output_s)
    output_s = re.sub(r'ZEIGEN ', r'ZEIGEN-BILDSCHIRM', output_s)
    output_s = re.sub(r'ZEIGEN$', r'ZEIGEN-BILDSCHIRM', output_s)

    output_s = re.sub(r'^([A-Z]) ([A-Z][+ ])', r'\1+\2', output_s)
    output_s = re.sub(r'[ +]([A-Z]) ([A-Z]) ', r' \1+\2 ', output_s)
    output_s = re.sub(r'([ +][A-Z]) ([A-Z][ +])', r'\1+\2', output_s)
    output_s = re.sub(r'([ +][A-Z]) ([A-Z][ +])', r'\1+\2', output_s)
    output_s = re.sub(r'([ +][A-Z]) ([A-Z][ +])', r'\1+\2', output_s)
    output_s = re.sub(r'([ +]SCH) ([A-Z][ +])', r'\1+\2', output_s)
    output_s = re.sub(r'([ +]NN) ([A-Z][ +])', r'\1+\2', output_s)
    output_s = re.sub(r'([ +][A-Z]) (NN[ +])', r'\1+\2', output_s)
    output_s = re.sub(r'([ +][A-Z]) ([A-Z]$)', r'\1+\2', output_s)
    output_s = re.sub(r'([A-Z][A-Z])RAUM', r'\1', output_s)
    output_s = re.sub(r'-PLUSPLUS', r'', output_s)
    
    output_s = re.sub(r'(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])', r'\1', output_s)
    output_s = re.sub(r'(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])', r'\1', output_s)
    output_s = re.sub(r'(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])', r'\1', output_s)
    output_s = re.sub(r'(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])', r'\1', output_s)

    output_s = re.sub(r'__EMOTION__', r'', output_s)
    output_s = re.sub(r'__PU__', r'', output_s)
    output_s = re.sub(r'__LEFTHAND__', r'', output_s)
    output_s = re.sub(r'__EPENTHESIS__', r'', output_s)
    
    #remove multiple spaces, strip trunc the whitespaces in the end
    # return re.split(r'\s+', output_s.strip())
    return output_s.split()

def merge_duplicate(l: List[str]):
    return [item[0] for item in groupby(l)]

if __name__ == "__main__":
    print(apply_regex(['Sdfsdf', 'S', 'H', '   ']))
    print('a      b  c '.split())
