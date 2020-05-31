import numpy


def torch_serialize_type(t):
    try:
        import torch

        _torch_types = {
            torch.uint8: 'UINT8',
            torch.int8: 'INT8',
            torch.int16: 'INT16',
            torch.float16: 'FLOAT16',
            torch.int32: 'INT32',
            torch.float32: 'FLOAT32',
            torch.int64: 'INT64',
            torch.float64: 'FLOAT64',
        }

        return _torch_types[t]
    except KeyError:
        raise KeyError("Torch tensors of type {} are not supported".format(t))
    except ImportError:
        raise ImportError('You need pytorch module for creating models')


_numpy_types = {
    numpy.uint8: 'UINT8',
    numpy.dtype('uint8'): 'UINT8',
    numpy.int8: 'INT8',
    numpy.dtype('int8'): 'INT8',
    numpy.uint16: 'UINT16',
    numpy.dtype('uint16'): 'UINT16',
    numpy.int16: 'INT16',
    numpy.dtype('int16'): 'INT16',
    numpy.float16: 'FLOAT16',
    numpy.dtype('float16'): 'FLOAT16',
    numpy.uint32: 'UINT32',
    numpy.dtype('uint32'): 'UINT32',
    numpy.int32: 'INT32',
    numpy.dtype('int32'): 'INT32',
    numpy.float32: 'FLOAT32',
    numpy.dtype('float32'): 'FLOAT32',
    numpy.uint64: 'UINT64',
    numpy.dtype('uint64'): 'UINT64',
    numpy.int64: 'INT64',
    numpy.dtype('int64'): 'INT64',
    numpy.float64: 'FLOAT64',
    numpy.dtype('float64'): 'FLOAT64',
    numpy.complex64: 'COMPLEX64',
    numpy.dtype('complex64'): 'COMPLEX64',
    numpy.complex128: 'COMPLEX128',
    numpy.dtype('complex128'): 'COMPLEX128',
}


def numpy_serialize_type(t):
    try:
        return _numpy_types[t]
    except KeyError:
        raise KeyError("Numpy tensors of type {} are not supported".format(t))
