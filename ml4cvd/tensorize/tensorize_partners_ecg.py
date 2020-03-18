import os
import re
import sys
import h5py
import errno
import array
import shutil
import struct
import base64
import struct
import hashlib
import argparse
import optparse
import numcodecs
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from dateutil.parser import parse
from joblib import Parallel, delayed
from xml.parsers.expat import ExpatError



# Define global variables
FPATH_DATA = os.path.expanduser('/data/partners_ecg')
fpath_xml = os.path.join(FPATH_DATA, 'xml_deid')
fpath_hd5 = os.path.join(FPATH_DATA, 'hd5_deid')


def clean_read_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Replace newline character with space
    text = re.sub(r'\n', ' ', text) 

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Replace two+ spaces with one space
    text = re.sub(r'  +', ' ', text)

    # Remove all leading and trailing whitespace
    text = text.strip()

    return text


def get_dirs_in_dir(fpath_dir):
    fpath_dirs = [f.path for f in os.scandir(fpath_dir) if f.is_dir()]
    fpath_dirs.sort()
    return fpath_dirs


def get_files_from_dir(fpath_dir, extension):
    # Ensure extension starts with period
    if extension[0] is not '.':
        extension = '.' + extension

    # Within directory (input string), find all XML files
    files = filter(lambda x: x.endswith(extension), os.listdir(fpath_dir))

    # Covert filter iterable to list
    files_list = list(files)

    # If list of XML files is empty, return 0
    if not files_list:
        return None

    # Sort list of XML files
    files_list.sort()

    # Prepend every XML file name with full path
    files_list[:] = [os.path.join(fpath_dir, fname)
                         for fname in files_list]

    #print('%s contains %d %s files'
    #     % (fpath_dir, len(files_list), extension) )

    return files_list


def hash_xml_fname(fpath_xml):
    with open(fpath_xml, 'rb') as f:
        bytes = f.read() # read entire file as bytes
        readable_hash = hashlib.sha256(bytes).hexdigest();
        return fpath_xml, readable_hash


def hash_xml_fname_dir(fpath_dir):

    # Within directory (input string), find all XML files
    xml_files = filter(lambda x: x.endswith('.xml'), os.listdir(fpath_dir))

    # Covert filter iterable to list of XML files
    xml_files_list = list(xml_files)

    # Sort list of XML files
    xml_files_list.sort()

    # Prepend every XML file name with full path
    fpath_xmls = [os.path.join(fpath_dir, xml_fname)
                  for xml_fname in xml_files_list]

    # Compute hashes in parallel
    hashes = Parallel(n_jobs=-1)(delayed(hash_xml_fname)(fpath_xml)
        for fpath_xml in fpath_xmls)

    return hashes


def get_voltage_from_xml(fpath_xml): 
    MuseXmlParser.start_element

    # Initialize parser object
    g_parser = MuseXmlParser()

    def start_element(name, attrs):
        g_parser.start_element(name, attrs)

    def end_element(name):
        g_parser.end_element(name)

    def char_data(data):
        g_parser.char_data(data)

    p = xml.parsers.expat.ParserCreate()
    p.StartElementHandler = start_element
    p.EndElementHandler = end_element
    p.CharacterDataHandler = char_data

    # Read the XML file and parse it
    p.ParseFile(open(fpath_xml, 'rb'))

    # Convert the data into a ZCG buffer
    g_parser.makeZcg()

    # Return voltage data as a dict of arrays
    voltage = g_parser.get_voltage()

    # Return voltage
    return voltage


def filter_ecg(df, fs=250, cutoff_low=0.5, cutoff_high=120):
    ecg_signal = mne.filter.notch_filter(df.values.T.astype(float),
                                         fs,
                                         60.,
                                         verbose=False)
    ecg_signal = mne.filter.filter_data(ecg_signal,
                                        fs,
                                        cutoff_low,
                                        cutoff_high,
                                        verbose=False)
    # Convert numpy.ndarray from float to int to save memory
    ecg_signal = ecg_signal.astype(int)
    return ecg_signal


def text_from_xml(xml_fpath):
    # Initialize empty dictionary in which to store text from the XML.
    ecg_data = dict()

    # Use lxml parser, which makes all tags lower case.
    with open(xml_fpath, 'r') as f:
        soup = BeautifulSoup(f, 'lxml')

    # If the XML is gibberish and un-parseable,
    # then soup.prettify() will return '' or False,
    # which is equivalent to "if not soup.prettify().
    # then we should not parse the contents, and return an empty dict()

    # If the XML is not gibberish, parse the contents.
    if soup.prettify():
        # Text is inside <RestingECG> tags, so ovewrite soup with that text.
        soup = soup.find('restingecg')
       
       # Define major tags.
        major_tags = ['patientdemographics',
                      'testdemographics',
                      'restingecgmeasurements',
                      'originalrestingecgmeasurements']

        # Loop through the major tags we want to extract.
        for tag in major_tags:

            # Ovewrite soup with contents in major tag.
            soup_of_tag = soup.find(tag)

            # Check if soup_of_tag is not None, otherwise find_all() throws an
            # error if called on a 'NoneType' object.
            if soup_of_tag is not None:

                print(soup_of_tag)
                # Find all child elements in subset.
                elements = soup_of_tag.find_all()

                # Iterate through child elements via list comprehension
                # and save the element key-value (name-text) to dict.
                [ecg_data.update({el.name : el.get_text()}) for el in elements]

        # Parse text of cardiologist read within <Diagnosis> tag and save to ecg_data dict.
        ecg_data.update({'diagnosis_md' : parse_soup_diagnosis(soup.find('diagnosis'))})

        # Parse text of computer read within <OriginalDiagnosis> tag and save to ecg_data dict
        ecg_data['diagnosis_computer'] = parse_soup_diagnosis(soup.find('originaldiagnosis'))

    # Return dict
    return ecg_data


def parse_soup_diagnosis(input_from_soup):

    parsed_text = ''

    if input_from_soup is None:
        return parsed_text
    else:
        parts = input_from_soup.find_all('diagnosisstatement')

        # Check for edge case where <diagnosis> </diagnosis> does not encompass
        # <DiagnosisStatement> sub-element, which results in parts being length 0
        if len(parts) > 0:
            for part in parts:

                # Create list of all <stmtflag> entries
                flags = part.find_all('stmtflag')

                # Isolate text from part
                text_to_append = part.find('stmttext').text

                # Initialize flag to ignore sentence, e.g. do not append it
                flag_ignore_sentence = False

                # If no reasons found, append
                if not flag_ignore_sentence:
                    # Append diagnosis string with contents within <stmttext> tags
                    parsed_text += text_to_append

                    endline_flag = False

                    # Loop through flags and if 'ENDSLINE' found anywhere, mark flag
                    for flag in flags:
                        if flag.text == 'ENDSLINE':
                            endline_flag = True

                    # If 'ENDSLINE' was found anywhere, append newline
                    if endline_flag:
                        parsed_text += '\n'

                    # Else append space
                    else:
                        parsed_text += ' '

            # Remove final newline character in diagnosis
            if parsed_text[-1] is '\n':
                parsed_text = parsed_text[:-1]

        return parsed_text


# Format input date from `mm-dd-yyyy` into `yyyy-mm`, or `yyyy-mm-dd`
# This is the ISO 8601 standard for date
def format_date(input_date, day_flag, sep_char='-'):

    date_iso = input_date[6:10] + sep_char \
                + input_date[0:2]

    if day_flag:
        date_iso = date_iso + sep_char + input_date[3:5]

    return date_iso


std_leads = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")


class XmlElementParser:
    """Abstract base class for a XML Parsing State. It contains methods for
    restoring the previous state and for tracking the character data between
    tags."""
    def __init__(self, old_State = None):
        self.__old_State = old_State
        self.__data_Text = ""
        
    def restoreState(self, context):
        """This method restores the previous state in the XML parser."""
        if self.__old_State:
            context.setState(self.__old_State)
            
    def clearData(self):
        """This method clears the character data that was collected during parsing"""
        self.__data_Text = ""
        
    def getData(self):
        """This method returns the character data that was collected during
        parsing and it strips any leading or trailing whitespace"""
        return str.strip(self.__data_Text)
        
    def start_element(self, name, attrs, context):
        print("""abstract method, called at the start of an XML element""")
        sys.exit(0)
        
    def end_element(self, name, context):
        print("""abstract method, called at the end of an XML element""")
        sys.exit(0)
        
    def char_data(self, data, context):
        """This method accumulates any character data"""
        self.__data_Text = self.__data_Text + data
        
        
class IdleParser(XmlElementParser):
    """State for handling the Idle condition"""
    def __init__(self):
        XmlElementParser.__init__(self)
    
    def start_element(self, name, attrs, context):
        if name == WaveformElementParser.Tag:
            context.setState(WaveformElementParser(self))
        
    def end_element(self, name, context):
        self.clearData()
        

class WaveformElementParser(XmlElementParser):
    """State for handling the Waveform element"""
    
    Tag = "Waveform"
    
    def __init__(self, old_State):
        XmlElementParser.__init__(self, old_State)
    
    def start_element(self, name, attrs, context):
        self.clearData()
        if name in (ElementParser.waveform_type_tag, ElementParser.voltage_base_tag):
            context.setState(ElementParser(self))
        elif name == LeadDataElementParser.Tag:
            context.setState(LeadDataElementParser(self))
            
    def end_element(self, name, context):
        if name == self.Tag:
            self.restoreState(context)

class LeadDataElementParser(XmlElementParser):
    """State for handling the LeadData element"""
    
    Tag = "LeadData"
    
    def __init__(self, old_State):
        XmlElementParser.__init__(self, old_State)
        
    def start_element(self, name, attrs, context):
        self.clearData()
        if name in (ElementParser.lead_id_tag, ElementParser.unit_tag,
            ElementParser.bit_tag, ElementParser.waveform_data_tag):
            context.setState(ElementParser(self))
        
    def end_element(self, name, context):
        if name == self.Tag:
            self.restoreState(context)

class ElementParser(XmlElementParser):
    """State for handling voltageBase Element, LeadID,
    LeadAmplitudeUnits, LeadAmplitudeUnitsPerBit, WaveformData
    and WaveFormType elements"""

    voltage_base_tag = "voltageBase"
    unit_tag = "LeadAmplitudeUnits"
    bit_tag = "LeadAmplitudeUnitsPerBit"
    lead_id_tag = "LeadID"
    waveform_type_tag = "WaveformType"
    waveform_data_tag = "WaveFormData"

    def __init__(self, old_State):
        XmlElementParser.__init__(self, old_State)

    def start_element(self, name, attrs, context):
        self.clearData()

    def end_element(self, name, context):
        if name == self.voltage_base_tag:
            self.restoreState(context)
            if context.found_Rhythm:
                context.setvoltageBase(self.getData())
        if name == self.unit_tag:
            self.restoreState(context)
            context.setUnits(str.strip(self.getData()))
        if name == self.bit_tag:
            self.restoreState(context)
            context.setAdu(float(str.strip(self.getData())))
        if name == self.lead_id_tag:
            self.restoreState(context)
            if context.found_Rhythm:
                #sys.stdout.write("   Lead %2s found..." % self.getData())
                context.addLeadId(self.getData())
        if name == self.waveform_type_tag:
            self.restoreState(context)
            if str.find(self.getData(), "Rhythm") >= 0:
                context.setRhythmFound(1)
                #print("ECG %s object found." % self.getData())
            else:
                context.setRhythmFound(0)
        if name == self.waveform_data_tag:
            self.restoreState(context)
            if context.found_Rhythm:
                #print("   Adding data for lead %2s." % context.lead_Id)
                context.addWaveformData(self.getData())
    
class MuseXmlParser:
    """This class is the parsing context in the object-oriented State pattern."""
    def __init__(self):
        self.ecg_Data = dict()
        self.ecg_leads = list()
        self.__state = IdleParser()
        self.found_Rhythm = 0
        self.voltage_Rate = 0
        self.adu_Gain = 1
        self.units = ""
        
    def setState(self, s):
        self.__state = s
        
    def getState(self):
        return self.__state
                        
    def setvoltageBase(self, text):
        if self.found_Rhythm:
            self.voltage_Rate = int(text)
            
    def setAdu(self, gain):
        self.adu_Gain = gain
        
    def setUnits(self, units):
        self.units = units
            
    def setRhythmFound(self, v):
        self.found_Rhythm = v
        
    def addLeadId(self, id):
        self.lead_Id = id
                
    def addWaveformData(self, text):
        self.ecg_Data[self.lead_Id] = base64.b64decode(text)
        self.ecg_leads.append(self.lead_Id)
        
    def start_element(self, name, attrs):
        """This function trackes the start elements found in the XML file with a
        simple state machine"""
        self.__state.start_element(name, attrs, self)                
            
    def end_element(self, name):
        self.__state.end_element(name, self)                
            
    def char_data(self, data):
        self.__state.char_data(data, self)                
        

    def makeZcg(self):
        """This function converts the data read from the XML file into a ZCG buffer
        suitable for storage in binary format."""

        # All of the leads should have the same number of voltage
        total_byte_chars = len(self.ecg_Data[self.ecg_leads[0]])

        # We have 2 bytes per ECG voltage, so make our buffer size n * DATAMUX
        self.zcg = array.array('d')

        # Append the data into our huge ZCG buffer in the correct order
        for t in range(0,total_byte_chars,2):
            for lead in self.ecg_leads:
                voltage = struct.unpack("h", bytes([self.ecg_Data[lead][t], self.ecg_Data[lead][t+1]]))
                self.zcg.append(voltage[0])        


    def get_voltage(self):
        # Multiply base voltage by the gain
        self.zcg = [x * self.adu_Gain for x in self.zcg]

        # Initialize dict of lists where L keys are ECG leads.
        # L is not always 8; sometimes you have right-sided leads (e.g. V3R).
        voltage = {new_list:[] for new_list in self.ecg_leads}

        # Loop through all indices in 1D voltage array, L leads at a time.
        # Correctly parsing the 1D array requires we parse all L leads.
        for i in range(0, len(self.zcg), len(self.ecg_leads)):

            # Initialize lead counter; this is necessary to increment through 
            # the 1D representation of a 2D array. For example, we start at
            # element i=0, then loop through element i+1, i+2, etc. until we
            # iterate through all 8 leads. Then we move to element i=8 (which
            # is the step size in the outermost for loop), and repeat.
            # 
            #  0 [5 ... ]
            #  1 [7 ... ]
            #  2 [2 ... ]
            #  3 [3 ... ]
            #  4 [1 ... ]
            #  5 [1 ... ]
            #  6 [4 ... ]
            #  7 [3 ... ]

            k = 0

            for lead in self.ecg_leads:

                voltage[lead].append(self.zcg[i + k])
                k = k + 1

        # Overwrite voltage dict with only standard 8 leads
        voltage = {lead:voltage[lead] for lead in std_leads}

        # Convert dict of lists into dict of arrays to enable vector math
        voltage = {key:np.array(val) for key,val in voltage.items()}

        # Perform vector math to obtain remaining 4 leads
        voltage['III'] = voltage['II'] - voltage['I']
        voltage['aVR'] = -1 * (voltage['I'] + voltage['II']) / 2
        voltage['aVL'] = voltage['I'] - voltage['II'] / 2
        voltage['aVF'] = voltage['II'] - voltage['I'] / 2

        return voltage 



def compress_data(hf, name, data, dtype, method='zstd',
                  compression_opts=19):
    # Define codec
    codec = numcodecs.zstd.Zstd(level=compression_opts)

    # If data is string, encode to bytes
    if dtype == 'str':
        data_compressed = codec.encode(data.encode())
        dsize = len(data.encode())
    else:
        data_compressed = codec.encode(data)
        dsize = len(data) * data.itemsize

    # Save data to hdf5
    dat = hf.create_dataset(name=name, data=np.void(data_compressed))

    # Set attributes
    dat.attrs['method']              = method
    dat.attrs['compression_level']   = compression_opts
    dat.attrs['len']                 = len(data)
    dat.attrs['uncompressed_length'] = dsize
    dat.attrs['compressed_length']   = len(data_compressed)
    dat.attrs['dtype'] = dtype
   

def decompress_data(data_compressed, dtype):
    codec = numcodecs.zstd.Zstd() 
    data_decompressed = codec.decode(data_compressed)
    if dtype == 'str':
        data = data_decompressed.decode()
    else:
        data = np.frombuffer(data_decompressed, dtype)
    return data


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parallel',
                        help='enables parallel processing with all cores',
                        action='store_true')
    args = parser.parse_args()
    return args


def _set_n_jobs_from_arg(args):
    # Scenario 1: user specifies True, or nothing (which defaults to True)
    if args.parallel:
        n_jobs = -1
    # Scenario 2: user specifies False
    else:
        n_jobs = 1
    return n_jobs


def create_fname_hd5(text_data, fpath_xml, sep_char='-'):
    # Fix malformed MRN by removing all non-numerical chars
    mrn = re.sub('[^0-9]', '', text_data['patientid'])

    fname = mrn \
            + sep_char \
            + utils.format_date(
                text_data['acquisitiondate'],
                sep_char=sep_char,
                day_flag=True) \
            + sep_char \
            + text_data['acquisitiontime'].replace(':', sep_char) \
            + sep_char \
            + utils.hash_xml_fname(fpath_xml)[1] \
            + '.hd5'
    return fname


def create_hd5_dir(fpath_hd5, yyyymm_str):
    # Determine yyyy-mm from acquisition date
    yyyymm = utils.format_date(yyyymm_str,
                               sep_char='-',
                               day_flag=False)
    # Determine full path to new hd5
    fpath_hd5_dir = os.path.join(fpath_hd5, yyyymm)

    # If hd5 dir does not exist, create it
    if not os.path.exists(fpath_hd5_dir):
        try:
            os.makedirs(fpath_hd5_dir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass

    # Return path to new hd5 directory
    return fpath_hd5_dir


def _get_max_voltage(voltage):
    max_voltage = 0
    for lead in voltage:
        if max(voltage[lead]) > max_voltage:
            max_voltage = max(voltage[lead])
    return max_voltage


def _remove_hd5_in_date_dir(fpath_xml_dir, fpath_hd5):
    # Isolate date dir name
    yyyymm = os.path.split(fpath_xml_dir)[1]

    # Determine path to associated hd5 directory
    fpath_hd5_dir = os.path.join(fpath_hd5, yyyymm)

    # If this directory exists, parse it
    if os.path.exists(fpath_hd5_dir):

        # Get list of hd5 files inside dir
        fpath_hd5_list = utils.get_files_from_dir(fpath_hd5_dir, 'hd5')

        # Check if hd5 files exist in the directory
        if fpath_hd5_list:

            # Iterate through hd5 files and delete them
            for fpath_hd5 in fpath_hd5_list:
                os.remove(fpath_hd5)


def _convert_xml_to_hd5(fpath_xml, fpath_hd5):
    # Set flag to check if we should convert to hd5
    convert = True
    
    # Isolate file name of this XML
    fname_xml = os.path.split(fpath_xml)[1]
    
    # Extract text data from XML into dict
    text_data = utils.text_from_xml(fpath_xml)

    # If XML is empty, remove the XML file and set convert to false
    if (os.stat(fpath_xml).st_size == 0 or not text_data):
        os.remove(fpath_xml)
        convert = False
        print('Conversion failed! XML is empty.')
    # XML is not empty, so try to get voltage
    else:
        # Extract voltage from XML
        try:
            voltage = utils.get_voltage_from_xml(fpath_xml)

            # If the max voltage value is 0, do not convert
            if _get_max_voltage(voltage) == 0:
                convert = False

        # If there is no voltage, or the XML is poorly formed,
        # the function will throw an exception, and we mark 'convert' to Fals.
        # However, ExpatError should be impossible to throw, since earlier
        # we catch XMLs filled with gibberish. 
        except (IndexError, ExpatError):
            print('Conversion failed! Voltage is empty or badly formatted.')
            convert = False

    # If convert is still true up to here, make the hd5
    if convert:

        # Define keys for cleaned reads
        key_read_md = 'diagnosis_md'
        key_read_pc = 'diagnosis_computer'
        key_read_md_clean = 'read_md_clean'
        key_read_pc_clean = 'read_pc_clean'
        
        # Clean cardiologist read
        read_md_clean = None
        if key_read_md in text_data.keys():
            read_md_clean = utils.clean_read_text(text=text_data[key_read_md])
        
        # Clean MUSE read
        read_pc_clean = None
        if key_read_pc in text_data.keys():
            read_pc_clean = utils.clean_read_text(text=text_data[key_read_pc])

        # Create new hd5 directory from acquisition date
        fpath_hd5_dir = create_hd5_dir(fpath_hd5=fpath_hd5,
                                     yyyymm_str=text_data['acquisitiondate'])
        
        # Create full path to new hd5 file based on acquisition date and time
        # e.g. /hd5/2004-05/mrn-date-time.hd5
        fname_hd5_new = create_fname_hd5(text_data=text_data,
                                       fpath_xml=fpath_xml, 
                                       sep_char='-')
       
        # Determine full path to new hd5 file
        fpath_hd5_new = os.path.join(fpath_hd5_dir, fname_hd5_new)

        # Create new hd5 file
        with h5py.File(fpath_hd5_new, 'w') as hf:

            # Iterate through each lead and save voltage array to hd5
            for lead in voltage.keys():
                compress_data(hf=hf,
                              name=lead,
                              data=voltage[lead].astype('int16'),
                              dtype='int16',
                              method='zstd')

            # Iterate through keys in extracted text data and save to hd5
            for key in text_data.keys():
                compress_data(hf=hf,
                              name=key,
                              data=text_data[key],
                              dtype='str',
                              method='zstd')

            # Save cleaned reads to hd5
            if read_md_clean:
                compress_data(hf=hf,
                              name=key_read_md_clean,
                              data=read_md_clean,
                              dtype='str',
                              method='zstd')

            if read_pc_clean:
                compress_data(hf=hf,
                              name=key_read_pc_clean,
                              data=read_pc_clean,
                              dtype='str',
                              method='zstd')

        print(f'Converted {fpath_xml} to {fpath_hd5_new}')
    return convert


def _convert_xml_to_hd5_wrapper(fpath_xml_dir, fpath_hd5, n_jobs):
    # Get list of full path to all XMLs in directory
    fpath_xml_list = utils.get_files_from_dir(fpath_xml_dir, 'xml')

    # DEBUG CODE
    fpath_xml_list = fpath_xml_list[0:10]

    print(fpath_xml_list)

    # If list of XML files is empty, return 0
    if not fpath_xml_list:
        print(fpath_xml_dir, 'does not contain any XMLs.')
        return 0

    # Convert in parallel
    converted = Parallel(n_jobs=n_jobs)(delayed(_convert_xml_to_hd5)(
                    fpath_xml, fpath_hd5) for fpath_xml in fpath_xml_list)

    return sum(converted)


def _get_mrn_date_time_from_fname(fname, sep_char):
    # Remove the path up until the file name
    mrn_date_time_hash = os.path.split(fname)[1]

   # Remove the hash from the end
    mrn_date_time = mrn_date_time_hash.rsplit(sep_char, 1)[0]

    return mrn_date_time


def _process_new_hd5(fpath_hd5_dir, sep_char):
    print(fpath_hd5_dir)

    # Initialize counter to track duplicates
    num_duplicates = 0
    
    # Get list of full path to all hd5s in directory
    fpath_hd5_list = utils.get_files_from_dir(fpath_hd5_dir, 'hd5')

    # If list is empty, return
    if not fpath_hd5_list:
        return num_duplicates

    # Sort the list
    fpath_hd5_list = sorted(fpath_hd5_list)

    # Isolate yyyy-mm-dd-hh-mm-ss from first hd5 file in list
    mrn_date_time_ref = _get_mrn_date_time_from_fname(fpath_hd5_list[0], sep_char)
    fpath_hd5_ref = fpath_hd5_list[0]

    # Rename this first file
    fpath_hd5_new = os.path.join(os.path.split(fpath_hd5_list[0])[0],
                                mrn_date_time_ref + '.hd5')
    os.rename(fpath_hd5_list[0], fpath_hd5_new)

    # Loop through remaining hd5 file names
    for fpath_hd5_this in fpath_hd5_list[1:]:
        mrn_date_time = _get_mrn_date_time_from_fname(fpath_hd5_this, sep_char)
       
        # If this mrn_date_time matches previous, we have a duplicate; delete it
        if mrn_date_time_ref == mrn_date_time:
            os.remove(fpath_hd5_this)
            num_duplicates += 1

        # If not, no duplicate, so rename file and update priors
        else:
            fpath_hd5_new = os.path.join(os.path.split(fpath_hd5_this)[0],
                                         mrn_date_time + '.hd5')
            os.rename(fpath_hd5_this, fpath_hd5_new)
            mrn_date_time_ref = mrn_date_time
            fpath_hd5_ref = fpath_hd5_this

    return num_duplicates


def _process_new_hd5_wrapper(fpath_hd5_dirs, n_jobs):
    duplicates = Parallel(n_jobs=n_jobs)(delayed(_process_new_hd5)(
        fpath_hd5_dir, sep_char='-') for fpath_hd5_dir in fpath_hd5_dirs)

    print('Removed %d duplicate hd5 files that escaped hash detection'
          % sum(duplicates) )

    return duplicates


if __name__ == '__main__':
    # Get user input args during python function call
    args = _get_args()

    # Specify n_jobs variable for parallel processing via joblib
    n_jobs = _set_n_jobs_from_arg(args)

    # Delete hd5 dirs from prior run
    '''
    print(f'Deleting existing hd5 files in {fpath_hd5}')
    if os.path.exists(fpath_hd5):
        fpath_hd5_dirs = utils.get_dirs_in_dir(fpath_hd5)
        for fpath_hd5_dir in fpath_hd5_dirs:
            shutil.rmtree(fpath_hd5_dir)
    else:
        os.mkdir(fpath_hd5)
    '''

    print(f'XML directories: {fpath_xml}')
    print(f'Destination hd5 files: {fpath_hd5}')

    # Get all XML directories
    fpath_xml_dirs = utils.get_dirs_in_dir(fpath_xml)

    # Sort XML directories from newest to oldest
    fpath_xml_dirs.sort(reverse=False)

    # DEBUG: focus on subset of XML dirs
    #fpath_xml_dirs = fpath_xml_dirs[0:1]

    # Initialize counter to track total number of converted hd5 files
    num_converted_hd5s = 0

    # Iterate through directories containing XML files
    for fpath_xml_dir in fpath_xml_dirs:
        # Remove all hd5 files from associated yyyy-mm directory
        _remove_hd5_in_date_dir(fpath_xml_dir, fpath_hd5)

        # Convert all XMLs inside the dir into hd5 files
        num_converted_hd5s += _convert_xml_to_hd5_wrapper(fpath_xml_dir,
                                                          fpath_hd5,
                                                          n_jobs)

    print(f'Converted {num_converted_hd5s} XML files to hd5') 

    # Get all hd5 directories
    fpath_hd5_dirs = utils.get_dirs_in_dir(fpath_hd5)

    print('Removing duplicate hd5 files, and removing hash from file names')

    # Iterate through directories containing hd5s in parallel
    duplicates = _process_new_hd5_wrapper(fpath_hd5_dirs, n_jobs)
