from typing import List
import collections
import os.path
import random
import json
import numpy
import h5py
from datetime import datetime
import sys
from ..utils.ingest import open_read_load, str_to_dtype, zstd_compress, zstd_decompress, xml_parse
from ..utils.ecg import process_waveform, ukbb_ecg_parse_timestamp
from ..utils.utils import add_hdf5_text_nonempty_int, add_text_nonempty_int, parse_missing_array,xml_extract_value_attributes

class Importer(object):
    def __init__(self, input_file = None, output_file = None, **kwargs):
        if input_file is not None: self._input_file = input_file
        else: self._input_file = None
        
        if output_file is not None: self._output_file = output_file
        else: self._output_file = None
        
        # Callback function for processing input files or building
        self._input_callback = collections.deque()
        self._build_callback = collections.deque()
        # JSON map from keys -> required values
        self._input_mapper = None
        self._built = False
        # Class type. Used in childen classes
        self._ctype = ""
        # Date time for operation. Will be set to now() when build() is called.
        self._datetime = datetime.now()
        # h5py.File
        self._file = None
        self._compress_all = True
        # Base modality (e.g. ECG, MRI, ...). This is required for all children instances
        self._modality = None
        # Do not process fields not captured in the mapper file
        self._mapper_only = True
        # Paths processed during mapping step. This should correspond to
        # a representative "path" in the input data. For example, this
        # should be the tag tree path for XML files.
        self._mapped_paths = []

        # These properties should be set by the user via keyword arguments.
        allowed_kwargs = {'preset', 
                          'compression',
                          }

        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)

    @property
    def input_file(self) -> str:
        return self._input_file

    @input_file.setter
    def input_file(self, file: str):
        if isinstance(file, str) == False:
            raise ValueError('Input values must be a string')
        
        self._input_file = file

    @property
    def output_file(self) -> str:
        return self._output_file

    @output_file.setter
    def output_file(self, file: str):
        if isinstance(file, str) == False:
            raise ValueError('Output value must be a string')
        
        self._output_file = file

    @property
    def built(self) -> bool:
        return self._built

    @property
    def ctype(self) -> bool:
        return self._ctype

    @property
    def input_callbacks(self) -> List[callable]:
        return self._input_callback

    @input_callbacks.setter
    def input_callbacks(self, functions: List[callable]):
        for i in functions:
            if callable(i) == False:
                raise ValueError('Must be a callable function')

        self._input_callback = functions

    @property
    def mapper_only(self) -> bool:
        return self._mapper_only

    @mapper_only.setter
    def mapper_only(self, state: bool):
        if isinstance(state, bool) == False:
            raise ValueError('Must be a boolean')

        self._mapper_only = state

    @property
    def build_callbacks(self) -> List[callable]:
        return self._build_callback

    @build_callbacks.setter
    def build_callbacks(self, functions: List[callable]):
        for i in functions:
            if callable(i) == False:
                raise ValueError('Must be a callable function')

        self._build_callback = functions

    @property
    def mapper(self) -> dict:
        return self._input_mapper

    @mapper.setter
    def mapper(self, mapper: str):
        if isinstance(mapper, str):
            try:
                self._input_mapper = json.loads(mapper)
            except ValueError as error:
                raise ValueError(f"Invalid JSON: {error}")

        elif isinstance(mapper, dict):
            try:
                self._input_mapper = json.loads(json.dumps(mapper))
            except ValueError as error:
                raise ValueError(f"Invalid JSON: {error}")
            except Exception:
                raise Exception('Could not parse dictionary -> JSON')
        else:
            raise ValueError('Invalid input format for mapper')

    def build(self):
        raise Exception("Cannot run build() on abstract base class")

    def automated_name(self, hex_length = 12) -> str:
        """Construct an automated output filename following the pattern
        ML4CVD-{base name}-{random hex}
        
        Keyword Arguments:
            hex_length {int} -- Number of hex characters to concatenate to the output name (default: {12})
        
        Raises:
            ValueError: No input file have been set
            ValueError: Input name is not a string
        
        Returns:
            str -- A string file path name
        """
        if self._input_file == None:
            raise ValueError('Cannot construct automatic name without first specifying the input file name')

        if isinstance(self._input_file, str) == False:
            raise ValueError('Input file name must be of type str')

        path, filename = os.path.split(self._input_file)
        return f"ML4CVD-{filename.split(os.extsep, 1)[0]}-{hex(random.getrandbits(128))[2:(hex_length+2)]}"


class XmlImporter(Importer):
    def __init__(self, input_file = None, **kwargs):
        super(XmlImporter, self).__init__(input_file, **kwargs)
        self._modality = 'ecg'
        self._hdf5_basepath = 'instance0'

    def _store_unique_dataset(self, path: str, field_name: str, data):
        """Save data in a HDF5 dataset.
        
        Arguments:
            path {[str]} -- Absolute path in directory
            field_name {[str]} -- Field name for dataset
            data -- Data to save in dataframe
        """
        if '/'.join([path, field_name]) not in self._mapped_paths:
            # Construct path in HDF5 file
            gname = '/'.join([self._modality, self._hdf5_basepath, path])
            g = self._file.require_group(name = gname)
            
            # Check if the target key exists in the list of keys
            if field_name in g.keys():
                # Update length
                g[field_name].attrs['depth'] += 1
                dat_name = field_name + '-' + str(g[field_name].attrs['depth'])

                # Store dataset
                dat = self._store_hdf5_dataset(g, dat_name, data, None, 19)

                # Lift over attributes
                # for a in g[field_name].attrs:
                #    dat.attrs[a] = g[field_name].attrs[a]
                
                dat.attrs['origin'] = (path + '/' + field_name).encode()

            # Key is not available in HDF5 file
            else:
                # Store dataset
                dat = self._store_hdf5_dataset(g, field_name, data, None, 19)
                dat.attrs['depth']  = 1
                dat.attrs['origin'] = (path + '/' + field_name).encode()

    def _store_hdf5_dataset(self, 
                            group: h5py.Group, 
                            dataset_name: str, 
                            data, 
                            mapper: dict, 
                            compression_level = 19):
        """Store data in a HDF5 dataset. Data will be compressed uisng Zstd if the
        compressed data is smaller compared to the input data. This cost/trade-off
        ignores the significant overhead of storing meta information in the HDF5
        format.
        
        Arguments:
            group {h5py.Group} -- Target HDF5 group to add new dataset to
            dataset_name {str} -- Name for the new dataset
            data {[type]} -- Data to store
            mapper {dict} -- Mapper dictionary if used
        
        Keyword Arguments:
            compression_level {int} -- Compression level for Zstd (if used) (default: {19})
        
        Raises:
            ValueError: [description]
        
        Returns:
            [type] -- [description]
        """
        if isinstance(group, h5py.Group) == False:
            raise ValueError("Group must be of type h5py.Group")

        # If the input data is a Numpy structure
        if type(data).__module__ == numpy.__name__:
            if mapper is not None: dtype = mapper['dtype']
            else: dtype = data.dtype

            ulen = data.size * data.itemsize
            compressed = zstd_compress(data.tobytes(), ulen, compression_level)
            clen = len(compressed)
            
            # Mutate data if the compressed size < uncompressed size
            if clen < ulen:
                # print(f"is numpy type: {data.size * data.itemsize} -> {len(compressed)} of type {data.dtype}")
                data = numpy.void(compressed)
            else:
                clen = ulen
            #     print(f"is numpy type: using uncompressed {data.size * data.itemsize}")

            dat = group.create_dataset(dataset_name, data = data)
            dat.attrs['len'] = data.size
            if clen < ulen: 
                dat.attrs['compression'] = 'zstd'
            else: 
                dat.attrs['compression'] = 'none'
            
            dat.attrs['uncompressed_size'] = ulen
            dat.attrs['compressed_size']   = clen
            dat.attrs['shape'] = data.shape
            dat.attrs['dtype'] = str(dtype)
        # If the input data is a string
        elif isinstance(data, str):
            ulen = len(data)
            compressed = zstd_compress(data.encode(), ulen, compression_level)
            clen = len(compressed)
            # Mutate data if the compressed size < uncompressed size
            # otherwise store as literal using special h5py ASCII encoding
            # type.
            if clen < ulen:
                dat = group.create_dataset(dataset_name, data = numpy.void(compressed))
                dat.attrs['dtype'] = "bstr"
                dat.attrs['compression'] = 'zstd'
            else:
                print(f"clen >= ulen. {clen} >= {ulen} for {data} for {dataset_name}")
                dat = group.create_dataset(dataset_name, data = numpy.void(data.encode()))
                dat.attrs['dtype'] = "str"
                dat.attrs['compression'] = 'none'
                clen = ulen
                print(f"{dataset_name}: {dat.attrs['dtype']}, {dat.attrs['compression']}, {bytes(dat[()]).decode()}")
            
            dat.attrs['len']   = len(data)
            dat.attrs['uncompressed_size'] = ulen
            dat.attrs['compressed_size']   = clen
        # All other cases
        else:
            ulen = len(data)
            compressed = zstd_compress(data, ulen, compression_level)
            clen = len(compressed)
            # Mutate data if the compressed size < uncompressed size
            if clen < ulen: data = numpy.void(compressed)
            else: clen = ulen
            dat = group.create_dataset(dataset_name, data = data)

            dat.attrs['len'] = len(data)
            if clen < ulen: 
                dat.attrs['compression'] = 'zstd'
                if mapper['dtype'] == 'str':
                    dat.attrs['dtype'] = "bstr"
                else:
                    dat.attrs['dtype'] = mapper['dtype']
            else: 
                dat.attrs['compression'] = 'none'
                dat.attrs['dtype'] = mapper['dtype']

            dat.attrs['uncompressed_size'] = ulen
            dat.attrs['compressed_size']   = clen
        
        return dat

    def _mapper(self, mapper, xml: collections.OrderedDict, path: str):
        """Depth-first search for target fields of interest in the JSON mappings
        file. Found string matches are replaced with their target primitive type 
        (dtype) and stored as {data: data, values: mapper JSON element}.
        
        Arguments:
            mapper -- Current subslice of JSON map
            xml {collections.OrderedDict} -- Parsed XML into a collections.OrderedDict
            path {str} -- Current path in tree
        
        Raises:
            ValueError: Input XML is not of type collections.OrderedDict, or 
            mapper is not provided
        """
        if mapper is None:
            raise ValueError('Cannot run mapper without JSON map')

        if isinstance(path, str) == False:
            raise ValueError("Tracked path must be of type str")

        for i,v in enumerate(mapper):
            if isinstance(mapper[v], dict):
                if isinstance(xml, list):
                    for l in xml:
                        if list(mapper[v].keys())[0] == 'dtype':
                            if mapper[v]['dtype'] != 'str':
                                # Cast string to numpy.ndarray with the desired dtype
                                l[v] = str_to_dtype(l[v], mapper[v]['dtype'])

                            # Callback functions are available
                            if 'callback' in mapper[v]:
                                for callback in mapper[v]['callback']:
                                    sp = callback.split('.')
                                    try:
                                        att = getattr(sys.modules['.'.join(sp[: len(sp) - 1])], sp[-1])
                                    except Exception:
                                        raise ValueError(f'Could not find subroutine: {callback}')
                                    
                                    l[v] = att(l[v])
                            
                            # Construct path in HDF5 file
                            if 'destination' in mapper[v]:
                                gname = '/'.join([self._modality, self._hdf5_basepath, mapper[v]['destination']])
                            else:
                                gname = '/'.join([self._modality, self._hdf5_basepath, path])

                            g = self._file.require_group(name=gname)
                            
                            # Check if the target key exists in the list of keys
                            if v in g.keys():
                                # Update length
                                g[v].attrs['depth'] += 1
                                dat_name = v + '-' + str(g[v].attrs['depth'])

                                # Store dataset
                                dat = self._store_hdf5_dataset(g, dat_name, l[v], mapper[v], 19)

                                # Lift over attributes
                                # for a in g[v].attrs:
                                #     dat.attrs[a] = g[v].attrs[a]                                        
                                
                                dat.attrs['origin'] = (path + '/' + v).encode()
                                self._mapped_paths.append(dat.attrs['origin'].decode())

                                if 'attrs' in mapper[v]:
                                    for m in mapper[v]['attrs']:
                                        if isinstance(m, dict) == False or len(m) != 1:
                                            raise ValueError('Attributes must be stored as (key, value)-tuple in a dict')
                                        
                                        ls = list(m)
                                        dat.attrs[ls[0]] = m[ls[0]]

                            # Key is not available in HDF5 file
                            else:
                                # Store dataset
                                dat = self._store_hdf5_dataset(g, v, l[v], mapper[v], 19)
                                dat.attrs['depth']  = 1
                                dat.attrs['origin'] = (path + '/' + v).encode()
                                self._mapped_paths.append(dat.attrs['origin'].decode())

                                if 'attrs' in mapper[v]:
                                    for m in mapper[v]['attrs']:
                                        if isinstance(m, dict) == False or len(m) != 1:
                                            raise ValueError('Attributes must be stored as (key, value)-tuple in a dict')
                                        
                                        ls = list(m)
                                        dat.attrs[ls[0]] = m[ls[0]]
                        else:
                            self._mapper(mapper, l, path)
                elif isinstance(xml[v], str):
                    if mapper[v]['dtype'] != 'str':
                        # Cast string to numpy.ndarray with the correct dtype
                        xml[v] = str_to_dtype(xml[v], mapper[v]['dtype'])

                    if 'callback' in mapper[v]:
                        for callback in mapper[v]['callback']:
                            sp = callback.split('.')
                            try:
                                att = getattr(sys.modules['.'.join(sp[:len(sp) - 1])], sp[-1])
                            except Exception:
                                raise ValueError(f'Could not find subroutine: {callback}')
                            
                            xml[v] = att(xml[v])
                    
                    # print(f'updated is now {xml[v]} at {path}')
                    # Construct path in HDF5 file
                    if 'destination' in mapper[v]:
                        gname = '/'.join([self._modality, self._hdf5_basepath, mapper[v]['destination']])
                    else:
                        gname = '/'.join([self._modality, self._hdf5_basepath, path])
                    
                    # print(f"final {xml[v]} at {path + '/' + v}")
                    g = self._file.require_group(name = gname)
                    dat = self._store_hdf5_dataset(g, v, xml[v], mapper[v], 19)
                    dat.attrs['depth']  = 1
                    dat.attrs['origin'] = (path + '/' + v).encode()
                    self._mapped_paths.append(dat.attrs['origin'].decode())

                    if 'attrs' in mapper[v]:
                        for m in mapper[v]['attrs']:
                            if isinstance(m, dict) == False or len(m) != 1:
                                raise ValueError('Attributes must be stored as (key, value)-tuple in a dict')
                            
                            ls = list(m)
                            dat.attrs[ls[0]] = m[ls[0]]
                else:
                    self._mapper(mapper[v], xml[v], path + '/' + v)

            else:
                # Should never happen
                print(f'========= {type(xml)}')
                exit(102)


    def _extract_xml_fields(self, xml: collections.OrderedDict, path: str):
        """Perform depth-first search of input XML object and populate target HDF5 file
        
        Arguments:
            xml {collections.OrderedDict} -- Input XML object of type collections.OrderedDict
            path {str} -- Current path in tree
        
        Raises:
            ValueError: If the input path is not a string
        """
        if isinstance(path, str) == False:
            raise ValueError("Tracked path must be of type str")

        if isinstance(xml, list):
            for i,v in enumerate(xml):
                if isinstance(xml[i], str) == False and isinstance(xml[i], collections.OrderedDict) == False:
                    self._extract_xml_fields(xml[i], '/'.join([path, v]))
                elif isinstance(xml[i], collections.OrderedDict):
                    for k in xml[i]:
                        if isinstance(xml[i][k], str):
                            # Store data
                            self._store_unique_dataset(path, k, xml[i][k])
                        else:
                            self._extract_xml_fields(xml[i][k], '/'.join([path, str(i), k]))
                else:
                    # Store data
                    self._store_unique_dataset(path, v, xml[i])
        else:
            if type(xml).__module__ == numpy.__name__:
                return

            for _,v in enumerate(xml):
                if isinstance(xml[v], str) == False and isinstance(xml[v], collections.OrderedDict) == False and type(xml[v]).__module__ != numpy.__name__:
                    self._extract_xml_fields(xml[v], '/'.join([path, v]))
                elif isinstance(xml[v], collections.OrderedDict):
                    for k in xml[v]:
                        if isinstance(xml[v][k], str):
                            # Store data
                            self._store_unique_dataset(path, k, xml[v][k])
                        else:
                            self._extract_xml_fields(xml[v][k], '/'.join([path, v, k]))
                elif type(xml[v]).__module__ == numpy.__name__:
                    pass
                else:
                    # Store data
                    self._store_unique_dataset(path, v, xml[v])

    def voltage_postprocessing(self, 
                            sample_rate: int = 500, 
                            wiggle_room: int = 50, 
                            peak_limit:  int = 20,
                            polynomial_degree: int = 6,
                            bandpass_low:   float = 0.1,
                            bandpass_high:  float = 30,
                            bandpass_order: int = 2):
        """Processes voltage data extracted from a MUSE XML file. Data arrays
        are detrended using a low-degree polynomial, filtered for noise using
        bandfilters, and R-peaks are predicted using an algebraic approach.
        
        Keyword Arguments:
            sample_rate {int} -- [description] (default: {500})
            wiggle_room {int} -- [description] (default: {50})
            peak_limit {int} -- [description] (default: {20})
            polynomial_degree {int} -- [description] (default: {6})
            bandpass_low {float} -- [description] (default: {0.1})
            bandpass_high {float} -- [description] (default: {30})
            bandpass_order {int} -- [description] (default: {2})
        
        Raises:
            IOError: [description]
            ValueError: [description]
            KeyError: [description]
            Exception: [description]
            KeyError: [description]
            KeyError: [description]
            KeyError: [description]
        """
        if self._file.__bool__() == False:
            raise IOError('HDF5 file not open')

        if 'latest' not in self._file[self._modality]:
            raise ValueError('Cannot automatically located target group to use (searched for "latest")')

        try:
            group = self._file[self._modality]['latest/data']
        except KeyError:
            raise KeyError('Data field has not been set')
        except Exception:
            raise Exception('Unknown exception')

        parsed_data = []

        for g in group:
            data = group[g][()]
            if group[g].attrs['compression'] == 'zstd':
                data = zstd_decompress(data)
                data = numpy.frombuffer(data, dtype = group[g].attrs['dtype'])
                parsed_data.append(data)
            else:
                parsed_data.append(numpy.frombuffer(data, dtype = group[g].attrs['dtype']))

        # p, axarrp = pyplot.subplots(6, 3, sharex=True)

        processed_data_len = []
        processed_data_b = bytes()
        peaks_len = []
        peaks_b   = bytes()

        for idx, d in enumerate(parsed_data):
            processed, peaks_x, peaks_y = process_waveform(d, sample_rate = 150)
            processed_data_len.append(len(processed))
            peaks_len.append(len(peaks_x))
            processed_data_b += processed.tobytes()
            peaks_b += peaks_x.tobytes()
            
        if 'processed' in self._file[self._modality]['latest']:
            raise KeyError('Processed group already exist')

        gname = '/'.join([self._modality,'latest/processed'])
        group = self._file.require_group(name = gname)
        if 'data' in group:
            raise KeyError('dataset data already exists')

        if 'peaks' in group:
            raise KeyError('dataset peaks already exists')

        # print(gname, peaks_len, processed_data_len)
        processed_data_b = zstd_compress(processed_data_b, 1, 19)
        dat = group.create_dataset('data', data = numpy.void(processed_data_b))
        dat.attrs['len'] = processed_data_len
        dat.attrs['dtype'] = 'int16'
        dat.attrs['compression'] = 'zstd'
        dat.attrs['uncompressed_size'] = len(processed_data_b)
        dat.attrs['compressed_size']   = len(processed)
        peaks_b = zstd_compress(peaks_b, 1, 19)
        group.create_dataset('peaks', data = numpy.void(peaks_b))

    def build(self, preset: str = None, compression: str = "zstd"):
        """Construct the target output HDF5 file.
        """
        if self.built == False:
            # Record time of build
            self._datetime = datetime.now()
            
            # Construct outpu filename if none is available
            if self._output_file == None or isinstance(self._output_file, str) == False:
                output_filename = self.automated_name()
            else:
                output_filename = self._output_file
            
            # Reset mapped paths
            self._mapped_paths = []

            # Open file
            self._file = h5py.File(f"{'/'.join([output_filename])}.h5", 'a')
            self._file.require_group(name = self._modality)
            if 'depth' in self._file.attrs:
                self._hdf5_basepath = 'instance' + str(self._file.attrs['depth'])
                self._file.attrs['depth'] += 1
                # TODO: Used at the moment for debugging
                print("here now dead")
                exit(1)
            else:
                self._file.attrs['depth'] = 1
                self._file.require_group(name = '/'.join([self._modality, self._hdf5_basepath]))

            using_preset = False
            if preset is not None:
                if preset == 'ukbb':
                    using_preset = True
                    self.input_callbacks = [open_read_load]
                    self.build_callbacks = [self.voltage_postprocessing]
                    self._input_mapper = None

                    if compression == "zstd":
                        self.input_callbacks.append(zstd_decompress)
                    self.input_callbacks.append(xml_parse)
                    # Parse UKBB data
                    self.input_callbacks.append(self.parse_ukbb_xml)

            # Iterate over input callbacks
            if len(self.input_callbacks):
                for k, c in enumerate(self.input_callbacks):
                    if k == 0: data = c(self._input_file)
                    else: data = c(data)
            else:
                data = open_read_load(self._input_file)

            # Perform optional mapping step
            if using_preset == False:
                if self._input_mapper is not None:
                    self._mapper(self._input_mapper, data, '')
                    if self._mapper_only == False:
                        self._extract_xml_fields(data, '')
                else:
                    self._extract_xml_fields(data, '')

            # Symbolic link to the latest dataset
            if 'latest' in self._file[self._modality]: del self._file[self._modality]
            self._file[self._modality]['latest'] = h5py.SoftLink(f'/{self._modality}/{self._hdf5_basepath}')
            # print(f"symlink {self._file[self._modality]['latest']}")

            # Iterate over build callbacks
            for b in self.build_callbacks:
                b(self)
        else:
            print('Already built')
        self._built = True

    # TODO: not part of importer but used during debugging
    def retrieve_processed_data(self):
        """Returns processed data as either a numpy.ndarray if all data is
        of equal lengths or an array of numpy.ndarrays if the shapes are
        different
        
        Raises:
            IOError: [description]
            ValueError: [description]
            ValueError: [description]
            KeyError: [description]
            Exception: [description]
            KeyError: [description]
            Exception: [description]
        
        Returns:
            numpy.ndarray -- Multidimensional Numpy array if the internal shapes are identical, or
            array -- Array of arrays if the internal shapes are different
        """
        if self._file.__bool__() == False:
            raise IOError('HDF5 file not open')

        if 'latest' not in self._file[self._modality]:
            raise ValueError('Cannot automatically located target group to use (searched for "latest")')

        if 'processed' not in self._file[self._modality]['latest']:
            raise ValueError('Cannot automatically located target dataset to use (searched for "processed")')

        try:
            group = self._file[self._modality]['latest/processed']
        except KeyError:
            raise KeyError('Data field has not been set')
        except Exception:
            raise Exception('Unknown exception')

        try:
            data = group['data']
        except KeyError:
            raise KeyError('Data field has not been set')
        except Exception:
            raise Exception('Unknown exception')

        lens = data.attrs['len']

        # TODO: check compression status first!
        if sum([k == lens[0] for k in lens]) == len(lens):
            return numpy.frombuffer(zstd_decompress(data[()]), dtype = data.attrs['dtype']).reshape((len(lens),lens[0]))
        else:
            ret = []
            offset = 0
            uncompressed_data = zstd_decompress(data[()])
            for l in lens:
                ret.append(numpy.frombuffer(uncompressed_data, offset = offset, count = l , dtype = data.attrs['dtype']))
                offset += l

            # if offset != len(data[()]):
            #     raise IOError(f"Incorrect decoding of data. Computed offset of {offset} but expected {len(data[()])}")

            return ret

    def parse_ukbb_xml(self, xml: collections.OrderedDict):
        """Custom function for parsing UKBB ECG XML data.
        
        Arguments:
            xml {collections.OrderedDict} -- Input parsed XML data
        """
        data = self._file.require_group(name = '/'.join([self._modality, self._hdf5_basepath]))

        # Store processed datetime
        data.create_dataset('recorded_timestamp', data = ukbb_ecg_parse_timestamp(xml['CardiologyXML']['ObservationDateTime']).__str__())
        data.create_dataset('dicom_uid', data = xml['CardiologyXML']['UID']['DICOMStudyUID'])

        # Store patient information from the ECG XML! This could be different from the
        # collective meta information for the UKBB cohort.
        target = xml['CardiologyXML']['PatientInfo']
        pinfo = data.create_group("patient_info")
        pinfo.create_dataset('PID',        data = int(xml_extract_value_attributes(target['PID'])['data']))
        pinfo.create_dataset('FamilyName', data = int(xml_extract_value_attributes(target['Name']['FamilyName'])['data']))
        pinfo.create_dataset('GivenName',  data = xml_extract_value_attributes(target['Name']['GivenName'])['data'])
        pinfo.create_dataset('Age',        data = int(xml_extract_value_attributes(target['Age'])['data']))
        pinfo.create_dataset('BornDay',    data = int(xml_extract_value_attributes(target['BirthDateTime']['Day'])['data']))
        pinfo.create_dataset('BornMonth',  data = int(xml_extract_value_attributes(target['BirthDateTime']['Month'])['data']))
        pinfo.create_dataset('BornYear',   data = int(xml_extract_value_attributes(target['BirthDateTime']['Year'])['data']))
        pinfo.create_dataset('Gender',     data = xml_extract_value_attributes(target['Gender'])['data'])
        pinfo.create_dataset('Race',       data = xml_extract_value_attributes(target['Race'])['data'])
        pinfo.create_dataset('Height',     data = int(xml_extract_value_attributes(target['Height'])['data']))
        pinfo.create_dataset('Weight',     data = float(xml_extract_value_attributes(target['Weight'])['data']))
        pinfo.create_dataset('PaceMaker',  data = xml_extract_value_attributes(target['PaceMaker'])['data'])
        
        # Diagnosis text
        diagnosis_text = str()
        if 'Interpretation' in xml['CardiologyXML']:
            try:
                target = xml['CardiologyXML']['Interpretation']['Diagnosis']['DiagnosisText']
                for _,j in enumerate(target):
                    if (j != '---'):
                        diagnosis_text += j + ' '

                data.create_dataset('diagnosis_text', data = diagnosis_text)
            except Exception:
                pass

            try:
                conclusion_text = str()
                target = xml['CardiologyXML']['Interpretation']['Conclusion']['ConclusionText']
                for _,j in enumerate(target):
                    # print(i, j)
                    if (j != '---'):
                        conclusion_text += j + ' '

                data.create_dataset('conclusion_text', data = conclusion_text)
            except Exception:
                pass

        # Add ECG measures
        target = xml['CardiologyXML']['RestingECGMeasurements']
        emeasure = data.create_group("resting_ecg_measurements")
        add_hdf5_text_nonempty_int(emeasure, 'VentricularRate', target, 'VentricularRate')
        add_hdf5_text_nonempty_int(emeasure, 'PQInterval', target, 'PQInterval')
        add_hdf5_text_nonempty_int(emeasure, 'PDuration', target, 'PDuration')
        add_hdf5_text_nonempty_int(emeasure, 'QRSDuration', target, 'QRSDuration')
        add_hdf5_text_nonempty_int(emeasure, 'QTInterval', target, 'QTInterval')
        add_hdf5_text_nonempty_int(emeasure, 'QTCInterval', target, 'QTCInterval')
        add_hdf5_text_nonempty_int(emeasure, 'RRInterval', target, 'RRInterval')
        add_hdf5_text_nonempty_int(emeasure, 'PPInterval', target, 'PPInterval')
        add_hdf5_text_nonempty_int(emeasure, 'PAxis', target, 'PAxis')
        add_hdf5_text_nonempty_int(emeasure, 'RAxis', target, 'RAxis')
        add_hdf5_text_nonempty_int(emeasure, 'TAxis', target, 'TAxis')
        try:
            emeasure.create_dataset('QRSNum', data = int(target['QRSNum']))
        except Exception:
            pass

        if 'MeasurementTable' in target:
            mtable = data.create_group("measurement_table")

            for _,j in enumerate(target['MeasurementTable']):
                if (j[0] == '@'): continue # skip attributes
                x = target['MeasurementTable'][j]
                if (isinstance(x, collections.OrderedDict)):
                    s = numpy.array(['nan' if p == ' ' or len(p) == 0 else p for p in x['#text'].split(',')])
                    # Cast char array to float array
                    # Numpy only have subroutines supporting NaNs when the primitive type is float
                    s = s.astype("float")
                    mtable.create_dataset(j, data = s)
                else:
                    # For the UKBiobank this value will always be 'LeadOrder'
                    mtable.create_dataset(j, data = x)

            mtable.create_dataset('POnset',  data = add_text_nonempty_int(xml_extract_value_attributes(target['POnset'])['data']))
            mtable.create_dataset('POffset', data = add_text_nonempty_int(xml_extract_value_attributes(target['POffset'])['data']))
            mtable.create_dataset('QOnset',  data = add_text_nonempty_int(xml_extract_value_attributes(target['QOnset'])['data']))
            mtable.create_dataset('QOffset', data = add_text_nonempty_int(xml_extract_value_attributes(target['QOffset'])['data']))
            mtable.create_dataset('TOffset', data = add_text_nonempty_int(xml_extract_value_attributes(target['TOffset'])['data']))

            # print(mtable.keys())

        if 'MedianSamples' in target:
            median_samples = data.create_group("median_samples")
            target = target['MedianSamples']
            median_samples.create_dataset('NumberOfLeads', data = target['NumberOfLeads'])
            median_samples.create_dataset('SampleRate',    data = add_text_nonempty_int(xml_extract_value_attributes(target['SampleRate'])['data']))
            median_samples.create_dataset('ChannelSampleCountTotal', data = target['ChannelSampleCountTotal'])
            median_samples.create_dataset('Resolution',    data = add_text_nonempty_int(xml_extract_value_attributes(target['Resolution'])['data']))
            median_samples.create_dataset('FirstValid',    data = add_text_nonempty_int(xml_extract_value_attributes(target['FirstValid'])['data']))
            median_samples.create_dataset('LastValid',     data = add_text_nonempty_int(xml_extract_value_attributes(target['LastValid'])['data']))

            # median_data = []

            for _,j in enumerate(target['WaveformData']):
                name = str()
                for k,l in enumerate(j):
                    if (k == 0):
                        name = j[l]

                s = j['#text'].replace("\t", "").replace("\n", "").split(',')
                s = numpy.array(['nan' if p == ' ' or len(p) == 0 else p for p in s])
                s = s.astype("int16")
                
                # Compress data using Blosc using Zstd. This approach
                # median_samples.create_dataset(name, data = s, **hdf5plugin.Blosc(cname='zstd', clevel=1, shuffle=hdf5plugin.Blosc.SHUFFLE))
                # Store data
                dat = self._store_hdf5_dataset(median_samples, name, s, None, 19)


        if 'f' in xml['CardiologyXML']:
            target       = xml['CardiologyXML']['VectorLoops']
            vector_loops = data.create_group("vector_loops")
            frontal      = vector_loops.create_group("Frontal")
            horizontal   = vector_loops.create_group("Horizontal")
            sagittal     = vector_loops.create_group("Sagittal")

            frontal.create_dataset(target['Frontal'][0][list(target['Frontal'][0].keys())[0]], data = parse_missing_array(target['Frontal'][0]['#text']))
            frontal.create_dataset(target['Frontal'][1][list(target['Frontal'][1].keys())[0]], data = parse_missing_array(target['Frontal'][1]['#text']))
            horizontal.create_dataset(target['Horizontal'][0][list(target['Horizontal'][0].keys())[0]], data = parse_missing_array(target['Horizontal'][0]['#text']))
            horizontal.create_dataset(target['Horizontal'][1][list(target['Horizontal'][1].keys())[0]], data = parse_missing_array(target['Horizontal'][1]['#text']))
            sagittal.create_dataset(target['Sagittal'][0][list(target['Sagittal'][0].keys())[0]], data = parse_missing_array(target['Sagittal'][0]['#text']))
            sagittal.create_dataset(target['Sagittal'][1][list(target['Sagittal'][1].keys())[0]], data = parse_missing_array(target['Sagittal'][1]['#text']))

            vector_loops.create_dataset('Resolution', data = add_text_nonempty_int(xml_extract_value_attributes(target['Resolution'])['data']))
            vector_loops.create_dataset('POnset',     data = add_text_nonempty_int(xml_extract_value_attributes(target['POnset'])['data']))
            vector_loops.create_dataset('POffset',    data = add_text_nonempty_int(xml_extract_value_attributes(target['POffset'])['data']))
            vector_loops.create_dataset('QOnset',     data = add_text_nonempty_int(xml_extract_value_attributes(target['QOnset'])['data']))
            vector_loops.create_dataset('QOffset',    data = add_text_nonempty_int(xml_extract_value_attributes(target['QOffset'])['data']))
            vector_loops.create_dataset('TOffset',    data = add_text_nonempty_int(xml_extract_value_attributes(target['TOffset'])['data']))
            vector_loops.create_dataset('ChannelSampleCountTotal', data = int(target['ChannelSampleCountTotal']))

        # Strip data
        target = xml['CardiologyXML']['StripData']
        # print(f"gname ===== {gname}")
        # strip_wave = self._file.require_group(name = gname)

        strip_data = data.create_group("strip_data")
        strip_wave = strip_data.create_group("data")
        

        for _,j in enumerate(target['WaveformData']):
            name = str()
            for k,l in enumerate(j):
                if (k == 0):
                    name = j[l]

            # Replace tabs and new lines with nothing (remove them)
            s = j['#text'].replace("\t", "").replace("\n", "").split(',')  
            # Convert empty values in CSV string into 'nan' strings. These will be
            # interpreted by Numpy as numpy.nan types.
            s = numpy.array(['nan' if p == ' ' or len(p) == 0 else p for p in s])
            # Cast the array into signed 16-bit integers.
            s = s.astype("int16")
            # Store data
            dat = self._store_hdf5_dataset(strip_wave, name, s, None, 19)

        strip_data.create_dataset('NumberOfLeads', data = int(target['NumberOfLeads']))
        strip_data.create_dataset('SampleRate',    data = add_text_nonempty_int(xml_extract_value_attributes(target['SampleRate'])['data']))
        strip_data.create_dataset('ChannelSampleCountTotal', data = int(target['ChannelSampleCountTotal']))
        strip_data.create_dataset('Resolution',    data = add_text_nonempty_int(xml_extract_value_attributes(target['Resolution'])['data']))

        if 'ArrhythmiaResults' in target:
            target = target['ArrhythmiaResults']
            arry_result = strip_data.create_group("strip_data")
            arry_result.create_dataset('BeatClass', data = numpy.array(target['BeatClass'], dtype="S"))

            times = []
            for _,j in enumerate(target['Time']):
                times.append(int(j['#text']))

            arry_result.create_dataset('Time', data = times)

        # Full disclosure information does not always exist
        # TODO: At the moment we are ignoring FullDisclosure data
        # try:
        #     disclose = _parse_missing_array(xml['CardiologyXML']['FullDisclosure']['FullDisclosureData']['#text'], pop_last=True, dtype="int16")
        #     #print(disclouse)
        # except KeyError:
        #     print("Full disclosure Key not available")
        # except:
        #     print("Other error type") # Usually an empty disclose tag (<FullDisclosure></FullDisclosure>)
        
        # strip_wave_processed = strip_data.create_group("processed")
        # ukbb_process_waveform(strip_wave, strip_wave_processed, filename = self.filename, plot = plot, plot_dir = plot_dir)

        # # Symbolic link to the latest dataset
        if 'data' in data:
            raise KeyError('Data group already exist')

        soft_path = '/' + '/'.join([self._modality, self._hdf5_basepath, 'strip_data', 'data'])
        data['data'] = h5py.SoftLink(soft_path)

    