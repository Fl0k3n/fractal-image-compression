from dataclasses import dataclass
from typing import BinaryIO

import numpy as np

from quadtree.common import ColoredQuadtreeImage, QuadtreeImage
from quadtree.decoder import QuadtreeDecoder
from quadtree.encoder import QuadtreeEncoder
from quadtree.serialization import QuadtreeDeserializer, QuadtreeSerializer
from utils import ChromaSubsampling, rgb_to_ycbcr, ycbcr_to_rgb


class RGBQuadtreeEncoder:
    def __init__(self, y_encoder: QuadtreeEncoder, cb_encoder: QuadtreeEncoder, cr_encoder: QuadtreeEncoder = None) -> None:
        self.y_encoder = y_encoder
        self.cb_encoder = cb_encoder
        self.cr_encoder = cr_encoder if cr_encoder is not None else cb_encoder

    def encode(self, rgb_img: np.ndarray, chroma_subsampling: ChromaSubsampling = "4:2:0") -> ColoredQuadtreeImage:
        ycbcr_img = rgb_to_ycbcr(rgb_img, chroma_subsampling)
        # TODO parallelize
        return ColoredQuadtreeImage(
            self.y_encoder.encode(ycbcr_img[0]),
            self.cb_encoder.encode(ycbcr_img[1]),
            self.cr_encoder.encode(ycbcr_img[2])
        )
    
class RGBQuadtreeDecoder:
    def __init__(self, y_decoder: QuadtreeDecoder, cb_decoder: QuadtreeDecoder = None, cr_decoder: QuadtreeDecoder = None) -> None:
        self.y_decoder = y_decoder
        self.cb_decoder = cb_decoder if cb_decoder is not None else y_decoder
        self.cr_decoder = cr_decoder if cr_decoder is not None else self.cb_decoder

    def decode(self, colored_img: ColoredQuadtreeImage) -> np.ndarray:
        y = self.y_decoder.decode(colored_img.encoded_Y)
        cb = self.cb_decoder.decode(colored_img.encoded_Cb)
        cr = self.cr_decoder.decode(colored_img.encoded_Cr)
        subsampling: ChromaSubsampling
        if y.shape == cb.shape == cr.shape:
            subsampling = "4:4:4"
        elif cb.shape == cr.shape:
            subsampling = "4:2:0"
        else:
            raise Exception("unsupported chroma subsampling")

        return ycbcr_to_rgb((y, cb, cr), subsampling)


class RGBQuadtreeSerializer:
    def __init__(self, serializer: QuadtreeSerializer) -> None:
        self.serializer = serializer
        self.file: BinaryIO = None 

    def serialize(self, img: ColoredQuadtreeImage, output: str | BinaryIO):
        self.file = output
        should_close = False
        try:
            if isinstance(output, str):
                should_close = True
                self.file = open(output, 'wb')
            self.serializer.serialize(img.encoded_Y, self.file)
            self.serializer.serialize(img.encoded_Cb, self.file)
            self.serializer.serialize(img.encoded_Cr, self.file)
        finally:
            if should_close:
                try:
                    self.file.close()
                except:
                    pass


class RGBQuadtreeDeserializer:
    def __init__(self, deserializer: QuadtreeDeserializer) -> None:
        self.deserializer = deserializer
    
    def deserialize(self, input: str | BinaryIO) -> ColoredQuadtreeImage:
        self.file = input
        should_close = False
        try:
            if isinstance(input, str):
                should_close = True
                self.file = open(input, 'rb')
            y = self.deserializer.deserialize(self.file)
            cb = self.deserializer.deserialize(self.file)
            cr = self.deserializer.deserialize(self.file)
            return ColoredQuadtreeImage(y, cb, cr)
        finally:
            if should_close:
                try:
                    self.file.close()
                except:
                    pass
