# Partners ECG
Code to work with MUSE 12-lead ECG files in XML and HD5 format.

## Table of contents
1. [Setup](https://github.com/mghcdac/partners-ecg#setup)  
2. [Extracting ECGs to XML](https://github.com/mghcdac/partners-ecg#extracting-ecgs-to-xml)  
3. [Data access](https://github.com/mghcdac/partners-ecg#data-access)
4. [ECG data fields](https://github.com/mghcdac/partners-ecg#ecg-data-fields)
5. [Source XMLs on `MAD3`](https://github.com/mghcdac/partners-ecg#source-xmls-on-mad3)
6. [Organizing XMLs and removing duplicates](https://github.com/mghcdac/partners-ecg#organizing-xmls-and-removing-duplicates)
7. [Converting files from XML to HDF5](https://github.com/mghcdac/partners-ecg#organizing-xmls-and-removing-duplicates)
8. [Extracting ECG metadata](https://github.com/mghcdac/partners-ecg#extracting-ecg-metadata)
9. [Other scripts](https://github.com/mghcdac/partners-ecg#other-scripts)

## Setup
```
$ git clone https://github.com/mghcdac/partners-ecg.git
$ cd partners-ecg
$ conda env create -f py37.yml
$ python script_to_run.py
```

## Extracting ECGs to XML
Extraction is the process of converting ECGs from the proprietary MUSE format and from within the MUSE database on a virtual machine into XML files that can be moved to a different compute environment for analysis.    

Instructions for the process are documented [here](https://github.com/mghcdac/MUSE-ECG/blob/master/other-documentation/how-to-extract-muse-ecg-data.md). 

## Data access
Access is obtained via the following steps:

1. Be added to the correct IRB.  
2. Obtain Partners Healthcare credentials.  
3. Request access to `MAD3` by contacting Brandon Westover (PI).  
3. On macOS, open Finder.app, connect to server (`⌘K`), and enter `smb://MAD3/MGH-NEURO-CDACS`.  
4. Enter your Partners Healthcare credentials and log in. 
5. Navigate to the path described above.   

## ECG data fields

Voltage is saved from XMLs as a dictionary of numpy arrays indexed by leads in the set `("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")`, e.g.:

```
voltage = {'I': array([0, -4, -2, ..., 7]),
          {'II': array([2, -9, 0, ..., 5]),
          ...
          {'V6': array([1, -4, -3, ..., 4]),
```

Every other element extracted from the XML by [utils/text_from_xml](https://github.com/mghcdac/partners-ecg/blob/master/utils.py) is returned as a string, even if the underlying primitive type is a number (e.g. age). Here are some of the more important elements:

```
acquisitiondate
atrialrate
dateofbirth
diagnosis_computer
diagnosis_md
ecgsamplebase
ecgsampleexponent
gender
heightin
location
locationname
overreaderfirstname
overreaderid
overreaderlastname
patientid
paxis
poffset
ponset
printerval
qoffset
qonset
qrscount
qrsduration
qtcfrederica
qtcorrected
qtinterval
race
raxis
taxis
toffset
ventricularrate
weightlbs
```

## Source XMLs on `MAD3`

3,382,146 ECGs are stored as XML files in `yyyy-mm` directories on `MAD3`, a [Partners storage server](https://rc.partners.org/it-services/storage-backup/mad3-faq):

```
MAD3/MGH-NEURO-CDAC/Projects/partners_ecgs/
.
├── brigham
│   ├── 1993-01
│   ├── ...
│   └── 2019-04
└── mgh
    ├── 1993-01
    ├── ...
    └── 2019-04
```

XML files are between 80-200 KB. The total size of `partners_ecg/mgh` is 405GB.  

The XMLs on `MAD3` are considered "source", or original copies, and should *not* be edited or manipulated. Users should only download a copy of these data onto their own compute resources for subsequent research.  


## Organizing XMLs and removing duplicates 

`1_organize_xml_into_yyyymm.py` moves XML files from a single directory into the appropriate yyyy-mm directory.


`2_remove_xml_duplicates.py` finds and removes exact duplicate XML files, as defined by every bit of two files being identical, determined via SHA-256 hashing. 


## Converting files from XML to HDF5

`python 3_convert_xml_to_hd5.py` extracts data from all XML files and saves as [HDF5 files](https://www.hdfgroup.org). 

One ECG from one XML is stored as one HDF5 file. 

This script is called with the `-p` or `--parallel` argument to parallelize conversion across all available CPUs.  


## Extracting ECG metadata

`4_extract_metadata_to_csv.py` iterates through every HDF5 file, identifies relevant data (e.g. MRN, diagnostic read, axes, intervals, age, gender, and race), and saves these data in a large CSV file:  

This CSV file will be used to construct a performant, queryable database to identify future cohorts for research projects.

## Other scripts

`utils.py` contains various utilities required for extracting text and voltage from XML files, processing strings, etc.

`query_cohort.py` loads a provided CSV file of MRNs, cross-references it against every HDF5 file, returns matching MRN matches and all accompanying ECG data, and saves specified metadata in a CSV file. This is extremely slow and will be sunset after the aformentioned database is created. 

`dataviz.py` contains data visualization scripts for plotting clinical-like 12-lead ECGs (in progress). 

Other scripts related to data labeling and training a deep learning model will be integrated into the [ml4cvd](https://github.com/broadinstitute/ml) codebase.  
