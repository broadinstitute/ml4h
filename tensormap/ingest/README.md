# Data ingestion

Clinical data is available in a multitude of different modalities such as electrocardiograms (ECGs), magnetic resonance imaging (MRI), accelerometer data, echocardiograms, angiograms, and so on. TensorMap abstracts the ingestion of data from several common modalities into a loosely defined schema using the HDF5 file format.

## Examples

## XML

### Presets

Several modalities have manually coded import procedures as presets. For example,importing ECGs recorded as part of the UK Biobank data cohort from compressed XMLs requires the `ukbb_ecg` preset and compression set to `zstd`.

```py
import tensormap.utils.ingest
# Bundled example data
cur_path = pathlib.Path(__file__).parent.absolute()
filename = str(cur_path) + '/../data/ingest/ecg/ukbb_scrubbed.xml.zst'
# XML importer
imp = tensormap.ingest.importer.XmlImporter(filename)
# Use an automatically generated name
imp.output_file = '/'.join(["./", imp.automated_name()])
# Ingest input data using
imp.build(preset = 'ukbb_ecg', compression = 'zstd')
```

Parsed waveform data can be retrieved using the `retrieve_processed_data()` function of `XmlImporter`. This subroutine returns either an array of arrays with different lengths or a `numpy.ndarray` if all individual waveform measurements are of equal length and the same primitive type. Internally, data will be decompressed and decoded, if required.

```py
>>> imp.retrieve_processed_data()
[array(array([ 50,  68,  69, ...,   2,  -8, -23], dtype = int16), ...]
```

### Default behavior

By default, when no import presets are used, data will be copied as literal strings from the input XML according to the XML hierarchy. For example, the XML field at `a/b/c/field` will be stored as `{root}/a/b/c/field` as a copy of the input field. Collisions in field name arising from the XML hierarchy will be resolved by appending an instance number. The following XML tree
```xml
<level>
    <field>a</field>
    <field>b</field>
    <field>c</field>
</level>
```
will be stored as `.../level/field`, `.../level/field-1`, `.../level/field-2`.

### Custom mappings

In order to provide support for importing files with no hard-coded preset in TensorMap, we provide the ability to use custom callback functions. These functions are executed in the following order:

1) Input callbacks for loading and preprocessing data: `input_callbacks`,
2) JSON mapper callback for controlling how select fields are processed and stored: `mapper`,
3) Post-procesing callbacks for modifying processed fields before closing the file-handle: `build_callbacks`.

Several input callbacks functions are defined but the most important are `open_read_load` that opens and reads a file into memory and `xml_parse` that parse an array of bytes into the required XML data structure. As an example:

```py
imp = tensormap.ingest.importer.XmlImporter(filename)
imp.output_file = '/'.join(["./", imp.automated_name()])
imp.input_callbacks = [tensormap.utils.ingest.open_read_load,
                       tensormap.utils.ingest.xml_parse]
```

will 1) first open and read a target file, and 2) parse those bytes into an XML object when `build()` is called.

The mapper callback takes as input a JSON file with the following structure:
```json
"field_name": {
    "dtype": "str", 
    "destination": "other/path/name", 
    "callback": [], 
    "attrs": [{"key": "value"}]
}
```

where only `dtype` is mandatory. The `field_name` must exactly match the target field name in the input XML file. The attribute `destination` can be used to divert the default path to another absolute path in the output HDF5 file. The `callback` functions must be functions that accepts input data and outputs the modified input data. The first decorator function must operate on string inputs. Lastly, `attrs` adds, or modifies, the attributes of the stored dataset.

A concrete example would be to map the waveform data in the MUSE format from base64-encoded 2-byte strings into `int16` vectors and divert that data to `{root}/data`:

```json
{
    "RestingECG":
    {
        "Waveform":
        {
            "LeadData": 
            {
                "WaveFormData": {
                    "dtype": "str", 
                    "destination": "data", 
                    "callback": ["tensormap.utils.ecg.muse_decode_waveform"], 
                    "attrs": [{"dtype": "int16"}]
                }
            }
        }
    }
}
```
Importantly, the mapper JSON structure must map exactly to the input XML hierarchy.
