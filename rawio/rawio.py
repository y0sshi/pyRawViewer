# ======================================================================
# Import
# ======================================================================
import struct
import numpy
import os
from enum import Enum
from imagecodecs import (
        jpegxl_encode, jpegxl_decode
        )

# ======================================================================
# Class
# ======================================================================
class ColorFilter(Enum):
    CLEAR = 0
    BAYER = 1
    QUAD  = 2

class Raw:
    def __init__(self, data = numpy.zeros((1, 1), dtype=numpy.uint16), color_filter = ColorFilter.CLEAR):
        self.data         = data
        self.color_filter = color_filter

    def readRaw(self, path_raw_in, color_filter = ColorFilter.CLEAR):
        self = Raw.new_fromRawFile(path_raw_in, color_filter)
        return self

    def writeBin(self, path_bin_out):
        height, width = self.get_shape()
        data_out      = numpy.concatenate([numpy.array([width, height]), self.data.flatten()])
        data_out.astype('<H').tofile(path_bin_out) #little endian, unsigned short(2byte / data)
        print(f"{path_bin_out} is saved.")
        return self

    def writeJxl(self, path_jxl_out: str, level=200, effort=1, lossless=True,bitspersample=16):
        '''
        save image in raw format

        :param path_jxl_out: output file name
        :param level: -
        :param effort: -
        :param lossless: -
        :param bitspersample: -
        :rtype None
        '''
        # bitspersample を None 以外に設定すると、
        # JXL_BIT_DEPTH_FROM_CODESTREAM の記録が変わるが
        # 指定しないか、arr.dtype が float の場合は
        # JXL_BIT_DEPTH_FROM_PIXEL_FORMAT になる。 
        # この bitspersample の値は arr.dtype により書き換わり使われない。
        encoded = jpegxl_encode(
                self.data,
                level=level,
                effort=effort,
                lossless=lossless,
                bitspersample=bitspersample)

        with open(path_jxl_out, 'wb') as fp:
            fp.write(encoded)

        return self

    def get_shape(self):
        return self.data.shape

    def get_width(self):
        return self.data.shape[1]

    def get_height(self):
        return self.data.shape[0]

    @classmethod
    def new(cls, width = 1, height = 1, color_filter = ColorFilter.CLEAR):
        return Raw (
                data         = numpy.zeros(shape=(height, width), dtype=numpy.uint16),
                color_filter = color_filter
                )

    @classmethod
    def new_fromBinFile(cls, path_bin_in, color_filter = ColorFilter.CLEAR):
        with open(path_bin_in, 'rb') as f:
            width   = struct.unpack_from('<H', f.read(2))[0]  #little endian, unsigned short(2byte / data)
            height  = struct.unpack_from('<H', f.read(2))[0]  #little endian, unsigned short(2byte / data)
            data_in = numpy.fromfile(f, '<H', width * height) #little endian, unsigned short(2byte / data)
            data    = numpy.reshape(data_in, [height, width]).astype(numpy.uint16)

        return Raw(
                data         = data,
                color_filter = color_filter
                )

    @classmethod
    def new_fromJxlFile(cls, path_jxl_in, color_filter = ColorFilter.CLEAR):
        """
        load raw format image file.
        coordinate convention is [r, c, ch]

        :param path_jxl_in: input file name
        :param color_filter: color_filter id
        :return -
        :rtype numpy.ndarray dtype=np.uint16
        """
        with open(path_jxl_in, 'rb') as img:
            data = jpegxl_decode(img.read())

        return Raw(
                data         = data,
                color_filter = color_filter
                )

    @classmethod
    def new_fromRawFile(cls, path_raw_in, color_filter = ColorFilter.CLEAR):
        """
        new from raw_file()
        """
        _, ext_raw_in = os.path.splitext(path_raw_in)
        
        if   ext_raw_in == ".bin":
            raw_in = Raw.new_fromBinFile(path_raw_in, color_filter)
        elif ext_raw_in == ".jxl":
            raw_in = Raw.new_fromJxlFile(path_raw_in, color_filter)
        else:
            print(f'[ERROR] *{ext_raw_in} format is not supported.')
            raw_in = Raw()

        return Raw(
                data         = raw_in.data,
                color_filter = raw_in.color_filter
                )

