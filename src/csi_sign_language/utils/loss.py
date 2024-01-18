from typing import Any
import torch 
import torch.nn as nn

class SelfDistill:
    def __init__(self, weights) -> None:
        #weigts: ctc_seq, ctc_conv, distill
        
        self.CTC_seq = nn.CTCLoss()
        self.CTC_conv = nn.CTCLoss()
        
        self.distll = None
        
    def __call__(self, output, target) -> Any:
        pass