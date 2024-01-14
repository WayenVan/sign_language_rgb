from typing import Any


class WarmUpLr:
    
    def __init__(self, initial_lr: float, target_lr: float, warm_up_rounds, min_lr=None, decay_factor=0.99) -> None:
        
        assert decay_factor <= 1.0
        assert initial_lr <= target_lr
        if min_lr is not None:
            assert min_lr <= target_lr
        
        self.warm_up_rounds = warm_up_rounds
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.min_lr = min_lr
        self.decay_factor = decay_factor
        
    def __call__(self, epoch) -> Any:
        if epoch <= self.warm_up_rounds - 1:
            return self.initial_lr + epoch * (self.target_lr - self.initial_lr) / self.warm_up_rounds
        
        if epoch >= self.warm_up_rounds:
            lr = self.target_lr * self.decay_factor ** (epoch - self.warm_up_rounds)
            if self.min_lr is not None and lr < self.min_lr:
                return self.min_lr
            return lr