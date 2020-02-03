import numpy
import h5py
from typing import Union, Callable

def exception_pass(func) -> Callable:
    """Decorator for ignoring exceptions
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    return wrapper


def xml_extract_value_attributes(data: Union[str, dict]) -> dict:
    """Extracts out the value and attributes of a parsed XML dictionary
    
    Arguments:
        data {str, dict} -- Input XML string or dictionary
    
    Raises:
        TypeError: Input is not of type str or dict
    
    Returns:
        dict -- Dictionary {data: val, attr: []} of value and attributes
    """
    if isinstance(data, str) == False and isinstance(data, dict) == False:
        raise TypeError("Input data must be of type str or dict")
    
    if len(data) == 0:
        return {"data": "", "attr": []}
    
    if isinstance(data, dict):
        ret = {"data": "", "attr": []}
        for v in data:
            if v == "#text":
                ret['data'] = data[v]
            elif v[0] == "@":
                ret['attr'].append({v[1:]: str(data[v])})
        
        return ret
    elif isinstance(data, str):
        return {"data": data, "attr": []}


# TODO: misleading name
def add_text_nonempty_int(key) -> numpy.array:
    """Internally scoped helper function for catching cases of empty 
    (key, value)-pairs. Modifies the target in-place.
    
    Arguments:
        key {[type]} -- Target JSON key used
    """
    if isinstance(key, str) == False:
        raise TypeError('Input key must be a string')

    if key != "":
        return numpy.array(key, dtype = "int32")
    else:
        return numpy.array(numpy.nan, dtype = "float")


# TODO: misleading name
def add_hdf5_text_nonempty_int(grp: h5py.Group, dataset_name: str, target, field_name: str) -> h5py.Group:
    try:
        data = target[field_name]
    except KeyError:
        return None

    return grp.create_dataset(dataset_name, data = add_text_nonempty_int(xml_extract_value_attributes(data)['data']))


def parse_missing_array(data: str, pop_last: bool = False, dtype: str = "int16") -> numpy.array:
    """Helper function to parse a comma-separated string of numerical values
    into a Numpy array.
    
    Arguments:
        data {str} -- Input CSV string
    
    Keyword Arguments:
        pop_last {bool} -- Delete trailing empty values (e.g. "1,2,3,,,") (default: {False})
        dtype {str} -- Desired Numpy dtype (default: {"int16"})
    
    Returns:
        numpy.ndarray -- A numpy.ndarray with the desired dtype
    """
    if isinstance(data, str) == False:
        raise TypeError("Input data must be of type str")
    
    s = data.replace("\t", "").replace("\n", "").split(',')
    if pop_last:
        while True:
            if len(s) == 0: break
            if len(s[-1]) != 0: break
            s.pop()

    s = numpy.array(['nan' if p == ' ' or len(p) == 0 else p for p in s])
    s = s.astype(dtype)
    
    return s

