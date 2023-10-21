from dataclasses import dataclass


@dataclass
class Domain:
    start_i: int
    start_j: int
    orientation: int
    rotation: int
    s: float
    o: float


class EncodedImage:
    def __init__(self, width: int, height: int, range_size: int) -> None:
        self.width = width
        self.height = height
        self.range_size = range_size
        self.domains: list[Domain] = []

                