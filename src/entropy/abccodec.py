from abc import ABC, abstractmethod

Code = int
CodeWidth = int

class EntropyEncoder(ABC):
    @abstractmethod
    def encode(self, val: int) -> tuple[Code, CodeWidth]:
        pass
        
class IdentityEncoder(EntropyEncoder):
    def __init__(self, bit_width: CodeWidth) -> None:
        self.bit_width = bit_width

    def encode(self, val: int) -> tuple[Code, CodeWidth]:
        return val, self.bit_width
    