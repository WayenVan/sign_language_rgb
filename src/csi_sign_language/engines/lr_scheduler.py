from typing import Any


class WarmUpLr:
    
    def __init__(self, initial_lambda: float, warm_up_rounds, min_lambda=None, decay_factor=0.99) -> None:
        
        assert decay_factor <= 1.0
        assert initial_lambda <= 1.0
        if min_lambda is not None:
            assert min_lambda <= 1.0
        
        self.warm_up_rounds = warm_up_rounds
        self.initial_lambda = initial_lambda 
        self.min_lambda = min_lambda
        self.decay_factor = decay_factor
        
    def __call__(self, epoch) -> Any:
        if epoch <= self.warm_up_rounds - 1:
            return self.initial_lambda + epoch * ( 1 - self.initial_lambda) / self.warm_up_rounds
        
        if epoch >= self.warm_up_rounds:
            lmda = self.decay_factor ** (epoch - self.warm_up_rounds)
            if self.min_lambda is not None and lmda < self.min_lambda:
                return self.min_lambda
            return lmda