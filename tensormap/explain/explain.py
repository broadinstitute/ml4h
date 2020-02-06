import h5py
import numpy
import numcodecs
import numpy
import glob
from typing import Union

import pdb

def str_trim_end_chars(data: str) -> str:
    return data


class ExplainStats():
    def __init__(self, **kwargs):
        self._is_string = False
        self._mean = 0
        self._n    = 0
        self._sum  = 0
        self._sum_squares = 0
        self._n_nans = 0
    
    def __str__(self) -> str:
        return(f"ExplainStats count {self._n}, with mean {self._mean}")

    def __add__(self, value):
        self._n += 1
        self._sum += value
        self._sum_squares += value * value


class Explain():
    def __init__(self, **kwargs):
        # Dictionary of dictionaries mapping paths -> types/tokens -> stats
        self._stats = dict()
        self._track_histogram = False
        self._tokenize_strings = False
        self._tokenize_token = " "
    
    def __str__(self) -> str:
        strings = []
        for s in self._stats:
            for k in self._stats[s]:
                strings.append(f"{s}-{k}: {self._stats[s][k].__str__()}")
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
                    zstd = numcodecs.zstd.Zstd()
                    decompressed = zstd.decode(node[()])

                    if self._stats[node.name].get(node.attrs['dtype']) == None:
                        self._stats[node.name][node.attrs['dtype']] = ExplainStats()

                    if node.attrs['dtype'] == "bstr":
                        print("is bstr comp")
                            
                    elif node.attrs['dtype'] == "str":
                        print("is str comp")
                    else:
                        # print(f"numpy compute stats: {node.attrs['dtype']}")

                        decompressed = numpy.frombuffer(decompressed, node.attrs['dtype'])
                    # if self._stats.get(node.name) == None:
                    #     self._stats[node.name] = dict()
                    #     self._stats[node.name][node.attrs['dtype']] = ExplainStats()

                # Is uncompressed
                elif node.attrs['compression'] == 'none':
                    if node.attrs['dtype'] == "bstr":
                        # Todo
                        if isinstance(node[()], numpy.void):
                            print('====>>>')
                            print(node.attrs.keys())
                            print(node[()])
                            print(node.name)
                            # print(bytes(node[()]).decode())
                            return 0

                        # print(f"is bstr uncomp with comp: {node.attrs['compression']}")
                        string = node[()]
                        print(f"type {type(string)}")
                        if isinstance(string, bytes):
                            string = string.decode()
                        elif isinstance(string, numpy.void):
                            print("is numpy void die")
                            print(string)
                            exit(1)

                        if self._tokenize_strings:
                            tokens = string.split(self._tokenize_token)
                            for t in tokens:
                                print(t)
                                if self._stats[node.name].get(t) == None:
                                    self._stats[node.name][t] = ExplainStats()
                                    self._stats[node.name][t]._n += 1
                                else:
                                    self._stats[node.name][t]._n += 1

                    elif node.attrs['dtype'] == "str":
                        string = node[()]
                        print(f"type {type(string)}")
                        if isinstance(string, bytes):
                            string = string.decode()
                        elif isinstance(string, numpy.void):
                            print("is numpy void die")
                            print(f"{node.name}: {string}")
                            print(node.attrs.keys())
                            for k in node.attrs:
                                print(f"attrs: {k}, {node.attrs[k]}")
                            
                            #pdb.set_trace()
                            string = bytes(string).decode()
                            #print(string)
                            #exit(1)
                        print(string)

                        if self._tokenize_strings:
                            tokens = string.split(self._tokenize_token)
                            for t in tokens:    
                                print(f"token {t}")
                                if self._stats[node.name].get(t) == None:
                                    self._stats[node.name][t] = ExplainStats()
                                    self._stats[node.name][t]._n += 1
                                else:
                                    self._stats[node.name][t]._n += 1
                    else:
                        print(numpy.frombuffer(node[()], node.attrs['dtype']))
                # Other unknown compression method
                else:
                    print("unknwon compression methid: {node.attrs['compression']}")
                    exit(1)
            # Attribute 'compression' have not been set
            # Assume that the data is uncompressed
            else:
                print('======compression not specified')
        # Current node is a group. Do nothing
        else:
            # print(f"group: {name} at {node.name}")
            pass
    
    def explain(self, input_file):
        # if isinstance(path, str) == False:
        #     raise TypeError('HDF5 path must be of type str')
        # recurse
        #input_file = glob.glob('ML4CVD-muse_ecg_deidentified_1*.h5')[1]
        with h5py.File(input_file, 'r') as f:
            f.visititems(self._visitor)
        return 1

