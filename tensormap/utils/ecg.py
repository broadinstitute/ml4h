import numpy
import numpy.polynomial.polynomial as poly
import base64
import struct
import array
import pandas
import collections
from scipy import signal as scisig
from scipy.signal import butter, lfilter, iirnotch
from datetime import datetime


_WINDOW_SEC = 0.160
_MIN_RR = 0.5 # compare with 0.33
_ARTICLE_SAMPLING_RATE = 5.0


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def detect(signal, rate):
    buffer, samples_delay = _filter_signal(signal, rate)
    buffer = _normalize(buffer)
    buffer = _compute_derivative(buffer)
    buffer = _normalize(buffer)
    buffer = [x * x for x in buffer]

    samples_window = round(_WINDOW_SEC * rate)
    samples_delay += samples_window / 2
    integrated = _window_integration(buffer, samples_window)

    min_rr_samples = round(_MIN_RR * rate)
    indices, _ = _thresholding(integrated, min_rr_samples)
    return [x - samples_delay for x in indices]


def _normalize(values, required_max=1.0):
    max_value = max(values)
    return [item / max_value * required_max for item in values]


def _low_pass_filter(signal):
    result = []
    for index, value in enumerate(signal):
        if index >= 1:
            value += 2 * result[index - 1]
        if index >= 2:
            value -= result[index - 2]
        if index >= 6:
            value -= 2 * signal[index - 6]
        if index >= 12:
            value += signal[index - 12]
        result.append(value)
    return result


def _high_pass_filter(signal):
    result = []
    for index, value in enumerate(signal):
        value = -value
        if index >= 1:
            value -= result[index - 1]
        if index >= 16:
            value += 32 * signal[index - 16]
        if index >= 32:
            value += signal[index - 32]
        result.append(value)
    return result


def _filter_signal(signal, rate):
    result = None
    delay = None
    if rate == _ARTICLE_SAMPLING_RATE:
        # fix: this filters work only for 200 Hz sampling rate
        buffer = _low_pass_filter(signal)
        result = _high_pass_filter(buffer)
        # In the paper delay is 6 samples for LPF and 16 samples for HPF
        # with sampling rate equals 200
        delay = 6 + 16
    else:
        nyq = 0.5 * rate
        lower = 5 / nyq
        upper = 15 / nyq
        b, a = scisig.butter(2, [lower, upper], btype="band")
        result = scisig.filtfilt(b, a, signal)
        delay = 0
    return result, delay


def _compute_derivative(signal):
    buffer = []
    max_value = 0.0
    for index in range(2, len(signal) - 2):
        value = (signal[index + 2] + 2 * signal[index + 1] -
                 signal[index - 2] - 2 * signal[index - 1])
        value /= 8.0
        if value > max_value:
            max_value = value
        buffer.append(value)
    return buffer


def _window_integration(signal, window_size):
    result = []
    value = 0
    for i, x in enumerate(signal):
        first = i - (window_size - 1)
        value += x / window_size
        if first > 0:
            value -= signal[first - 1] / window_size
        result.append(value)
    return result


def _thresholding(integrated, min_rr_samples):
    spki = 0
    npki = 0
    peaks = []
    threshold1 = spki
    th1_list = []
    i = 0
    while i < len(integrated) - 2:
        i += 1
        th1_list.append(threshold1)
        peaki = integrated[i]
        if peaki < integrated[i - 1] or peaki <= integrated[i + 1]:
            continue

        if peaki <= threshold1:
            npki = 0.875 * npki + 0.125 * peaki
        else:
            spki = 0.875 * spki + 0.125 * peaki

        threshold1 = npki + 0.25 * (spki - npki)
        # threshold2 = 0.5 * threshold1

        if peaki > threshold1:
            if not peaks:
                peaks.append(i)
            elif i - peaks[-1] >= min_rr_samples:
                peaks.append(i)
            elif integrated[peaks[-1]] < peaki:
                peaks[-1] = i
    return peaks, th1_list


def muse_decode_waveform(data, dtype: str = "int16") -> numpy.array:
    """Decode waveform data from MUSE-exported XMLs. This data
    is shorts (16-bit) values that have been base-64 encoded for
    a textual representation.
    
    Arguments:
        data {Any} -- Encoded waveform data
    
    Keyword Arguments:
        dtype {str} -- Numpy dtype for for the return array (default: {"int16"})
    
    Raises:
        TypeError: [description]
        TypeError: [description]
    
    Returns:
        numpy.array -- Array of decoded waveform data
    """
    if isinstance(dtype, str) == False:
        raise TypeError('dtype must be a string')
        
    try:
        data = base64.b64decode(data)
    except Exception:
        raise TypeError('Could not base-64 decode input data')

    zcg = array.array('d')
    # Iterate over pairs of bytes
    for t in range(0,len(data),2):
        sample = struct.unpack("h", data[t:t+2])
        zcg.append(sample[0])

    return numpy.array(zcg, dtype = dtype)


def waveform_string_toarray(data: str, dtype: str = "int16", csv_delimiter: str = ',') -> numpy.ndarray:
    """Convert a delimited string of waveform data into a Numpy ndarray of parsed
    values.
    
    Arguments:
        data {str} -- Input delimited string
    
    Keyword Arguments:
        dtype {str} -- Numpy dtype for output data (default: {"int16"})
        csv_delimiter {str} -- Delimiter character (default: {','})
    
    Raises:
        TypeError: [description]
        TypeError: [description]
    
    Returns:
        numpy.ndarray -- Parsed Numpy array
    """
    if isinstance(data, str) == False:
        raise TypeError(f'Input data must be of type str. Given: {type(data)}')

    if isinstance(dtype, str) == False:
        raise TypeError('dtype must be of type str')

    # Replace tabs and new lines with nothing (remove them)
    s = data.replace("\t", "").replace("\n", "").split(csv_delimiter)  
    # Convert empty values in CSV string into 'nan' strings. These will be
    # interpreted by Numpy as numpy.nan types.
    s = numpy.array(['nan' if p == ' ' or len(p) == 0 else p for p in s])
    # Cast the array into signed 16-bit integers.
    s = s.astype(dtype)
    return s


def process_waveform(data: numpy.ndarray,
                     sample_rate: int = 500, 
                     wiggle_room: int = 50, 
                     peak_limit:  int = 20,
                     polynomial_degree: int = 6,
                     bandpass_low: float = 0.1,
                     bandpass_high: float = 30,
                     bandpass_order: int = 2):
    
    x_local = numpy.arange(1, len(data)+1, 1, dtype = data.dtype)
    x_new   = numpy.linspace(x_local[0], x_local[-1], num = len(x_local))
    
    # Fit a low-degree polynomial
    coefs = poly.polyfit(x_local, data, polynomial_degree)
    ffit  = poly.polyval(x_new, coefs)
    # Remove data drift using the fitted line
    y_base_adjusted = data - ffit

    # Attempt to denoise using a Butterworth filter
    processed = butter_bandpass_filter(y_base_adjusted, bandpass_low, bandpass_high, sample_rate, order = bandpass_order)
    processed = numpy.array(processed, dtype = data.dtype)

    # Detect peaks
    peaks = numpy.array(detect(processed, sample_rate), dtype = "int")

    # Hyperparameter for X-axis shifting in index-steps to find maximum peak in window
    for k,j in enumerate(peaks):
        slices = processed[max(0,int(j)-wiggle_room):min(len(processed),int(j)+wiggle_room)]
        maxpos = (max(0,j)-wiggle_room) + numpy.where(abs(slices) == numpy.amax(abs(slices)))[0][0]
        peaks[k] = maxpos
        
    # Remove peaks that are smaller than the desired peak limit
    # computed as |y| >= threhold.
    peaks = peaks[numpy.absolute(processed[peaks]) >= peak_limit]

    return processed, peaks, processed[peaks]

def ukbb_ecg_parse_timestamp(xml: collections.OrderedDict) -> datetime:
    obs_time = pandas.DataFrame(columns = ['Hour', 'Minute', 'Second', 'Day', 'Month', 'Year'])
    obs_time['Hour']   = [int(xml['Hour'])]
    obs_time['Minute'] = [int(xml['Minute'])]
    obs_time['Second'] = [int(xml['Second'])]
    obs_time['Day']    = [int(xml['Day'])]
    obs_time['Month']  = [int(xml['Month'])]
    obs_time['Year']   = [int(xml['Year'])]
    datetime_str = f"{obs_time['Month'][0]:02d}/{obs_time['Day'][0]:02d}/{str(obs_time['Year'][0])[2:4]} {obs_time['Hour'][0]:02d}:{obs_time['Minute'][0]:02d}:{obs_time['Second'][0]:02d}"
    datetime_object = datetime.strptime(datetime_str, '%m/%d/%y %H:%M:%S')
    return datetime_object

