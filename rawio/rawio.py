# ======================================================================
# Import
# ======================================================================
import struct
import numpy
from enum import Enum

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

    def readBin(self, path_bin, color_filter = ColorFilter.CLEAR):
        self = new_fromBinFile(path_bin, color_filter)
        return self

    def writeBin(self, path_bin):
        height, width = self.get_shape()
        data_out      = numpy.concatenate([numpy.array([width, height]), self.data.flatten()])
        data_out.astype('<H').tofile(path_bin) #little endian, unsigned short(2byte / data)
        print(f"{path_bin} is saved.")
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
    def new_fromBinFile(cls, path_bin, color_filter = ColorFilter.CLEAR):
        with open(path_bin, 'rb') as f:
            width   = struct.unpack_from('<H', f.read(2))[0]  #little endian, unsigned short(2byte / data)
            height  = struct.unpack_from('<H', f.read(2))[0]  #little endian, unsigned short(2byte / data)
            data_in = numpy.fromfile(f, '<H', width * height) #little endian, unsigned short(2byte / data)
            data    = numpy.reshape(data_in, [height, width]).astype(numpy.uint16)

        return Raw(
                data         = data,
                color_filter = color_filter
                )

