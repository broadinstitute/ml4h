import collections
import numcodecs
import xmltodict
import numpy


def open_read_load(file: str, mode: str = 'rb') -> bytes:
    """Open a file and read its entire contents
    
    Arguments:
        file {str} -- Target file name
        mode {str} -- Mode to open file handle in
    
    Raises:
        TypeError: If the input value is not an input string
        IOError: Could not open the file
        Exception: All other potential errors
    
    Returns:
        bytes -- Returns a (binary) string
    """
    if isinstance(file, str) == False:
        raise TypeError('Input value must be a string path')

    try:
        with open(file, mode) as r:
            return r.read() 
    except IOError:
        raise IOError(f'Failed to open file: {file}')
    except Exception:
        raise Exception('Other error')


def zstd_decompress(data: bytes) -> bytes:
    """Decompress a stream of bytes using Zstandard.
    
    Arguments:
        data {bytes} -- Input stream of bytes
    
    Raises:
        ValueError: If data failed to decompress.
    
    Returns:
        bytes
    """
    codec = numcodecs.zstd.Zstd()
    try:
        uncompressed = codec.decode(data)
    except Exception:
        raise ValueError('Failed to decompress input data using Zstd')

    return uncompressed


def zstd_compress(data, length = None, level = 19) -> bytes:
    if length == None:
        try:
            if len(data) == 0:
                return bytes()
        except Exception:
            raise ValueError(f'Cannot find length of type: {type(data)}')
    else:
        if length == 0:
            return bytes()

    codec = numcodecs.zstd.Zstd(level = level)
    try:
        compressed = codec.encode(data)
    except Exception:
        raise ValueError('Failed to compress input data using Zstd')

    return compressed


def xml_parse(data: bytes) -> collections.OrderedDict:
    """Converts an input byte stream of XML data into a parsed ordered dictionary.
    
    Arguments:
        data {bytes} -- Input data stream
    
    Raises:
        ValueError: Failed to parse XML -> dictionary
    
    Returns:
        collections.OrderedDict
    """
    try:
        return xmltodict.parse(data)
    except ValueError:
        raise ValueError('Failed to parse XML -> dictionary')


def str_to_dtype(data: str, dtype: str, sep: str = None) -> numpy.ndarray:
    """Converts an input string into a numpy.ndarray.
    
    Arguments:
        data {str} -- Input data string
        dtype {str} -- Target primitive type (numpy.dtype)
    
    Keyword Arguments:
        sep {str} -- Delimiter for string arrays (default: {None})
    
    Raises:
        TypeError: If the input types are not string
        ValueError: If the provided dtype is illegal
    
    Returns:
        numpy.ndarray
    """
    if isinstance(data, str) == False:
        raise TypeError('Input type must be of type str')

    if isinstance(dtype, str) == False:
        raise TypeError('Dtype must be of type str')

    allowed_dtypes = {'int',
                    'int16',
                    'int32',
                    'int64',
                    'uint',
                    'uint16',
                    'uint32',
                    'uint64',
                    'float',
                    'float32'
                    }

    if dtype not in allowed_dtypes:
        raise ValueError(f"Illegal dtype: {dtype}")

    # print(sep == None)
    if sep is not None:
        return numpy.fromstring(data, dtype=dtype, sep=sep)
    else:
        return numpy.array(data, dtype=dtype)


