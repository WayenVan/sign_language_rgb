import abc
from typing import List, Tuple

class IPostProcess(abc.ABC):
    
    abc.abstractmethod
    def process(hyp: List[List[str]], gt: List[List[str]]) -> Tuple[any, any]:
        pass
