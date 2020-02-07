import h5py
import numpy
import numcodecs
import numpy
import glob
from typing import Union


class ExplainStats():
    """Internal class for storing summary statistics for both numeric and
    string input values. Updating this structure should be done primarily
    through the overloaded add operator (`__add__`) not the add equality
    operator (+=).
    """
    def __init__(self, **kwargs):
        self._is_string = False
        self._mean = 0
        self._n    = 0
        self._sum  = numpy.float64(0)
        self._sum_squares = numpy.float64(0)
        self._n_nans = 0
        self._min = numpy.float64(0)
        self._max = numpy.float64(0)
        self._shapes = dict()
    
    def __str__(self) -> str:
        if self._is_string:
            return(f"{self._n}")
        else:
            if self._sum != 0:
                self._mean = self._sum/self._n
            else:
                self._mean = 0
            return(f"{self._n}, {self._sum}, {self._sum_squares}, {self._min}, {self._max}, {self._mean}")

    def update_vector(self, vector: numpy.ndarray):
        if self._shapes.get(vector.shape) == None:
            self._shapes[vector.shape] = 1
        else:
            self._shapes[vector.shape] += 1

        for v in vector:
            self.__add__(v)

    def __add__(self, value):
        if type(value).__module__ == numpy.__name__:
            if value.shape != ():
                return self.update_vector(value)
            else:
                self._min = min(self._min, value)
                self._max = max(self._max, value)
                self._n += 1
                self._sum += value
                
                # Convert smaller primitive types to signed 64-bit values to
                # prevent overflowing/underflowing issues.
                if value.itemsize < 8 and str(value.dtype) != "float32":
                    value.astype('int64')
                    self._sum_squares += numpy.int64(value)**2
                else:
                    self._sum_squares += value * value
        elif isinstance(value, str):
            self._is_string = True
            self._n += 1
        else:
            self._min = min(self._min, value)
            self._max = max(self._max, value)
            self._n += 1
            self._sum += value
            self._sum_squares += value * value


class Explain():
    def __init__(self, **kwargs):
        # Dictionary of dictionaries mapping paths -> types/tokens -> stats
        self._stats     = dict()
        self._callbacks = dict()
        self._track_histogram  = False
        self._tokenize_strings = False
        self._tokenize_token   = " "
        self._tokenize_only    = False
    
    def __str__(self) -> str:
        strings = []
        for s in self._stats:
            for k in self._stats[s]:
                strings.append(f"{s}: {'STRING ' if self._stats[s][k]._is_string else ''}{k} {self._stats[s][k].__str__()}")

        return '\n'.join(strings)
    
    def _visitor(self, name: str, node: Union[h5py.Group, h5py.Dataset]):
        """Visitor function used in a depth-first search of a given HDF5 file. 
        This function is applied to each node in the target HDF5 file.
        Groups are ignored. Datasets are decompressed and used for updating summary statistics
        as follows: path -> type/token/string -> statistics. Statistics are stored in self._stats.
        
        Arguments:
            name {str} -- Name of node
            node {Union[h5py.Group, h5py.Dataset]} -- h5py node
        """
        if isinstance(node, h5py.Dataset):
            if self._stats.get(node.name) == None:
                self._stats[node.name] = dict()

            # Attribute 'compression' have been set
            if node.attrs.get('compression') != None:
                # Is compressed with Zstd
                if node.attrs['compression'] == 'zstd':
                    # Ascertain correctness
                    if node.attrs.get('dtype') == None:
                        raise KeyError("Mandatory attribute 'dtype' is not set for this compressed dataset.")

                    zstd = numcodecs.zstd.Zstd()
                    decompressed = zstd.decode(node[()])

                    if node.attrs['dtype'] == "bstr":
                        decompressed = bytes(decompressed).decode()
                        if self._tokenize_only == False:
                            if self._stats[node.name].get(decompressed) == None:
                                self._stats[node.name][decompressed] = ExplainStats()        
                                # Increment full string
                                self._stats[node.name][decompressed] + (decompressed)

                        if self._tokenize_strings:
                            tokens = decompressed.split(self._tokenize_token)
                            print(len(tokens))
                            if len(tokens) > 1:
                                for t in tokens:
                                    print(t)
                                    if self._stats[node.name].get(t) == None:
                                        self._stats[node.name][t] = ExplainStats()
                                    self._stats[node.name][t] + (t)
                            
                    elif node.attrs['dtype'] == "str":
                        raise TypeError('Illegal relationship: Zstd compression and storage type as "str"')
                    else:
                        if self._stats[node.name].get(node.attrs['dtype']) == None:
                            self._stats[node.name][node.attrs['dtype']] = ExplainStats()

                        decompressed = numpy.frombuffer(decompressed, node.attrs['dtype'])
                        self._stats[node.name][node.attrs['dtype']] + (decompressed)

                # Is uncompressed
                elif node.attrs['compression'] == 'none':
                    if node.attrs['dtype'] == "bstr":
                        string = node[()]
                        if isinstance(string, bytes):
                            string = string.decode()
                        elif isinstance(string, numpy.void):
                            string = bytes(string).decode()

                        if self._tokenize_only == False:
                            if self._stats[node.name].get(string) == None:
                                self._stats[node.name][string] = ExplainStats()        
                                # Increment full string
                                self._stats[node.name][string] + (string)


                        if self._tokenize_strings:
                            tokens = string.split(self._tokenize_token)
                            print(len(tokens))
                            if len(tokens) > 1:
                                for t in tokens:
                                    print(t)
                                    if self._stats[node.name].get(t) == None:
                                        self._stats[node.name][t] = ExplainStats()
                                    self._stats[node.name][t] + (t)

                    elif node.attrs['dtype'] == "str":
                        string = node[()]
                        if isinstance(string, bytes):
                            string = string.decode()
                        elif isinstance(string, numpy.void):
                            string = bytes(string).decode()

                        if self._tokenize_only == False:
                            if self._stats[node.name].get(string) == None:
                                self._stats[node.name][string] = ExplainStats()        
                                # Increment full string
                                self._stats[node.name][string] + (string)

                        if self._tokenize_strings:
                            tokens = string.split(self._tokenize_token)
                            print(len(tokens))
                            if len(tokens) > 1:
                                for t in tokens:    
                                    print(t)
                                    if self._stats[node.name].get(t) == None:
                                        self._stats[node.name][t] = ExplainStats()
                                    self._stats[node.name][t] + (t)
                    else:
                        if self._stats[node.name].get(node.attrs['dtype']) == None:
                            self._stats[node.name][node.attrs['dtype']] = ExplainStats()

                        self._stats[node.name][node.attrs['dtype']] + numpy.frombuffer(node[()], node.attrs['dtype'])
                        # print(numpy.frombuffer(node[()], node.attrs['dtype']))
                # Other unknown compression method
                else:
                    raise ValueError(f"Unknown compression type: {node.attrs['compression']}")
            # Attribute 'compression' have not been set
            # Assume that the data is uncompressed
            else:
                # TODO: Add support for uncompressed datasets without meta fields
                print(f"Currently unsupported: compression type is not specified for field {node.name}")
        # Current node is a group. Do nothing
        elif isinstance(node, h5py.Group):
            # print(f"group: {name} at {node.name}")
            pass
        else:
            raise TypeError('Illegal input type in visitor pattern')
    
    def explain(self, input_file):
        with h5py.File(input_file, 'r') as f:
            f.visititems(self._visitor)


