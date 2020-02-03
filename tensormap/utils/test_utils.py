import tensormap.utils.utils
import pytest
import numpy


def test_text_nonempty_valid():
    x = {"#text": "45"}
    assert numpy.isnan(tensormap.utils.utils.add_text_nonempty_int(x['#text'])) == False


def test_text_nonempty_not_string_dict():
    x = {"#text": 45}
    with pytest.raises(TypeError):
        tensormap.utils.utils.add_text_nonempty_int(x['#text'])


def test_text_nonempty_float():
    with pytest.raises(ValueError):
        tensormap.utils.utils.add_text_nonempty_int("45.1")


def test_text_nonempty_not_string():
    with pytest.raises(TypeError):
        tensormap.utils.utils.add_text_nonempty_int(45)


def test_text_nonempty_empty():
    assert numpy.isnan(tensormap.utils.utils.add_text_nonempty_int("")) == True


def test_parse_missing_array_valid():
    x = "1,2,3,4,5,6"
    assert numpy.array_equal(tensormap.utils.utils.parse_missing_array(x, pop_last = False, dtype = "int16"), 
            numpy.array([1,2,3,4,5,6], dtype="int16")) == True


def test_parse_missing_array_invalid_input():
    x = [1,2,3,4,5,6]
    with pytest.raises(TypeError):
        tensormap.utils.utils.parse_missing_array(x, pop_last = False, dtype = "int16")


def test_parse_missing_array_valid_typecheck():
    x = "1,2,3,4,5,6"
    assert str(tensormap.utils.utils.parse_missing_array(x, pop_last = False, dtype = "int16").dtype) == "int16"


def test_parse_missing_array_valid_last():
    x = "1,2,3,4,5,6,"
    assert numpy.array_equal(tensormap.utils.utils.parse_missing_array(x, pop_last = True, dtype = "int16"), 
            numpy.array([1,2,3,4,5,6], dtype="int16")) == True


def test_parse_missing_array_missing_float():
    x = "1,2,3,4,5,6,,"
    assert numpy.allclose(tensormap.utils.utils.parse_missing_array(x,pop_last=False,dtype="float"), 
                numpy.array([1,2,3,4,5,6,numpy.nan, numpy.nan], dtype="float"), equal_nan=True) == True


def test_parse_missing_array_missing_multi_trim():
    x = "1,2,3,4,5,6,,,,,,"
    assert numpy.array_equal(tensormap.utils.utils.parse_missing_array(x,pop_last=True,dtype="float"), numpy.array([1,2,3,4,5,6], dtype="float")) == True


def test_parse_missing_array_missing_multi_trim_empty():
    x = ","
    assert numpy.array_equal(tensormap.utils.utils.parse_missing_array(x,pop_last=True,dtype="float"), numpy.array([], dtype="float")) == True


def test_parse_missing_array_missing_multi_trim_notlast():
    x = "1,2,3,4,5,6,,,,,,5"
    assert numpy.allclose(tensormap.utils.utils.parse_missing_array(x,pop_last=True,dtype="float"), 
                numpy.array([1,2,3,4,5,6,numpy.nan,numpy.nan,numpy.nan,numpy.nan,numpy.nan,5], dtype="float"), equal_nan=True) == True


def test_xml_extract_value_attributes_str_emtpy():
    assert tensormap.utils.utils.xml_extract_value_attributes("") == {"data": "", "attr": []}


def test_xml_extract_value_attributes_str():
    x = "bananaman"
    assert tensormap.utils.utils.xml_extract_value_attributes(x) == {"data": x, "attr": []}


def test_xml_extract_value_attributes_str_attr1():
    x = {"#text": "bananaman", "@color": "yellow"}
    assert tensormap.utils.utils.xml_extract_value_attributes(x) == {"data": x['#text'], "attr": [{"color": "yellow"}]}


def test_xml_extract_value_attributes_str_attr2():
    x = {"#text": "bananaman", "@color": "yellow", "@length": 45}
    assert tensormap.utils.utils.xml_extract_value_attributes(x) == {"data": x['#text'], "attr": [{"color": "yellow"}, {"length": "45"}]}


def test_xml_extract_value_attributes_str_attr2_data():
    x = {"#text": "bananaman", "@color": "yellow", "@length": 45}
    assert tensormap.utils.utils.xml_extract_value_attributes(x)['data'] == "bananaman"


####### DECORATORS ##########
from contextlib import contextmanager

@contextmanager
def not_raises(exception):
  try:
    yield
  except exception:
    raise Exception


@tensormap.utils.utils.exception_pass
def bad_function(value):
    value['45'] = 12


def test_exception_pass_pass():
    with not_raises(Exception):
        bad_function("string")

