# Partners ECG
Extracting and tensorizing MUSE 12-lead ECGs

## Table of Contents
1. [Extracting ECGs to XML](#extracting-ecgs-to-xml)
2. [Automating ECG Extraction by MRN Search](#automating-ecg-extraction-by-mrn-search)
3. [Organizing XMLs and Removing Duplicates](#organizing-xmls-and-removing-duplicates)
4. [Tensorizing XMLs to HDF5](#tensorizing-xmls-to-hdf5)
5. [ECG Data Structure](#ecg-data-structure)
6. [Extracting ECG Metadata](#extracting-ecg-metadata)
7. [MUSE Virtual Machine Setup](#muse-virtual-machine-setup)

## Extracting ECGs to XML
### 1. Open the MUSE Editor
1. Connect to a Virtual Machine with MUSE using [chrome remote desktop](https://remotedesktop.google.com/access).
2. Log in to user account.
   1. MGH: `MuseAdmin` with password `Muse!Admin`.
   2. BWH: `musebkgnd` with password `Muse!Bkgnd`.
3. If your VM's trial of Windows is expired, see [MUSE Virtual Machine Setup](#muse-virtual-machine-setup).
4. Once logged in, go to the Desktop and open Services.
5. Select `MUSE` in the list that appears, click on "Start" in the left panel, and then close "Services".
6. Go to the Desktop and open "MUSE Editor".

### 2. Configure MUSE Editor
1. Open File Explorer and go to `C:\`. Create a new folder (Ctrl + Shift + N) and name it `export`.
2. Go to MUSE Editor and go to Device Setup: System -> Setup (Ctrl + Shift + P). The MUSE software will close and a new window will open.
3. Add a new folder by clicking the "New" icon in the top bar and selecting "Folder".  
![MUSE New Export Folder](images/MUSE-new-export-folder.png)  
Alternatively, from the top horizontal navigation bar: Action -> New -> Folder. A new window will appear called "Device Properties - Folder".
4. Fill in the "Device Name" field with the name `00export`. (`00` helps put the folder at the top of the print list).
5. Fill in the "Destination" field with the full path to the folder `C:\export`. Make sure you capitalize the `C`!
6. Enter `xml` for "File Extension" and select `XML` for "Output Type".  
7. Check all three boxes under "Output Options" including `Convert Statement Codes to Text`, `Include Measurement Matrix`, and `Include Waveforms`.
8. Click "OK".
9. Set up keyboard shortcuts. From the top menu bar -> Tools -> Options -> Shortcut Keys.
   1. Set "List > Select All Items" to `Ctrl + A`
   2. Set "Test > Print" to `Ctrl + P`

### 3. Search for ECGs in MUSE Editor
There are two ways of searching for ECGs: [by MRN](#a-search-by-mrnpatientid) or [by Date Range](#b-search-by-date-range)

#### a. Search by MRN/PatientID
1. Go to MUSE Editor and go to Edit/Retrieve: System -> Edit/Retrieve (Ctrl + Shift + E). The MUSE software will close and a new window will open.
2. Enter a MRN in the "Patient ID" field of the Test/Order box in the bottom left hand corner. Click "Search".  
![MUSE Editor Test Search Box](images/MUSE-test-search-field.png)
3. A list of ECGs should populate in the box at the bottom of the screen. Highlight this entire list (click the first row, hold Shift, click the last row). 
4. Once the entire list is selected, click the "Print Test" button in the top menu bar.  
![MUSE Editor Print Test Button](images/MUSE-print-test-button.png)  
A new window will appear called "Select Device and Formatting Options".

#### b. Search by Date Range
1. Go to MUSE Editor and go to Database Search: System -> Database Search (Ctrl + Shift + D). The MUSE software will close and a new window will open.
2. In the left vertical navigation bar called "Searches and results", click on "Scheduled searches". Double-click any search to open "Template search setup".  
![MUSE Editor Database Search](images/MUSE-database-search.png) 
3. Change the "Report Title" to the date range of interest e.g. `2005-03` for the entire month of March 2005.
4. Under "Date Field", select `Acquisition Date`.  
5. Under "Scheduling", select `Run Once Now`.  
6. Select the appropriate date range (arrow keys move between month/day/year, tab moves between start and end dates).  
7. Click "Ok" (or "Apply" then "Close").
8. In the left vertical navigation bar called "Searches and results", click on "Search results".
9. Wait for the search result to show up. Refresh the view by clicking refresh in the top menu bar.  
![MUSE Editor Search Results Refresh](images/MUSE-search-results-refresh.png)
10. Double click your search result. A new window will appear called "Search Results".
11. If there are `> 5000` records, check the box "Display full result set".
12. Click "Print all tests" and then "Yes". A new window will appear called "Select Device and Formatting Options".

### 4. Export ECGs from MUSE Editor
1. From "Select Device and Formatting Options", select the device you set up in section 2 as the output folder, it will probably be the first item in that list and already highlighted.
2. Set "Number of Copies" to `1`.
3. Set "Priority" to `Normal`
4. Set "Formatting" to `Use the default...`
5. Uncheck "Temporary Device"
6. Leave "Recipient Name" blank.
7. Click "OK". This should now export the ECG as XML to the folder from section 2.
8. If exporting `> 100` ECGs, MUSE Editor will likely freeze. This is normal.
9. Wait for ECGs to finish exporting. If the "Date modified" column in File Explorer for the folder shows the folder was last modified `> 1 hour` ago, the ECGs are likely done exporting.
10. Move the XML files to a data store, like MAD3 (`\\MAD3\MGH-NEURO-CDAC\Projects\partners_ecg\`) or a Partners DropBox (it is easier to download DropBox Desktop on the VM than to upload via the web browser to DropBox).

## Automating ECG Extraction by MRN Search
It is possible to automate the extraction of ECGs from MUSE Editor using [AutoHotKey](https://www.autohotkey.com/), a macro software. There are steps to setup the environment such that the macro can run and caveats to its use.

At a high level, the macro uses key bindings and window focus to accomplish its automation. The macro can queue ECG extractions at a rate of 600 MRNs/hour. The actual time it takes for the ECGs to finish exporting to XML will be longer and is dependent on the number of ECGs.

**To use the macro:**
1. [Download and install AutoHotKey](https://www.autohotkey.com/).
2. [Download and install Sublime Text](https://www.sublimetext.com/).
3. [Download the macro script](MUSE_search_mrn.ahk).
   1. Double click the script to enable the macro (this does not trigger the macro yet).
   2. Optionally, compile the script into a `.exe` file using AutoHotKey. This makes the macro portable without the need for AutoHotKey to be installed. Double click the executable file to enable the macro.
4. Do step: [Configure MUSE Editor](#2-configure-muse-editor)
5. Maximize the MUSE Editor window.
6. In MUSE Editor, go to the Edit/Retrieve screen (Ctrl + Shift + E). In the bottom left panel, set the ECG type (e.g. `Resting ECG`).
7. Close all other windows except for MUSE Editor.
8. Open Sublime Text. Open a list of MRNs. For example, `mrn.csv` might look like:
```
mrn
000000001
000000002
000000003
```
9. Place the cursor before the first MRN. If the pipe character `|` is the cursor:
```
mrn
|000000001
000000002
000000003
```
10. Make sure Sublime Text is the window in focus and MUSE Editor is the window directly underneath.
11. Start the macro with `Alt + R`.
12. The macro ends when there are no more items in the MRN list:
```
mrn
|
```

**Caveats:**
1. This only runs in Windows. However, MUSE also exists in a Windows VM so this is likely fine.
2. The slowest and most unpredictable part of the search process is waiting for the print screen of MUSE Editor to unfreeze. The macro detects pixel changes on the screen to tell when the window unfreezes. However, the pixel area is hard coded to a maximized MUSE Editor window. This feature is open to discussion and change.
3. Additionally, waiting for MUSE Editor to return all the ECGs for a given MRN also takes an unpredictable amount of time. Currently, the macro is hard coded to wait 1 second. A patient is not likely to have `> 200` ECGs which MUSE Editor can find in less than 1 second. However, there are potential edge cases which may break the macro here.
4. Other windows may be open on the virtual machine, however at the start of the macro, Sublime Text must be focused and MUSE Editor must be the window directly below Sublime Text. This is because the macro uses `Alt + Tab` to switch between windows.
5. Because the window focus must be Sublime Text and MUSE Editor, it is not safe to connect directly to the virtual machine running the macro, as the connection will likely pop up a new window and window focus will be lost.
6. A recommendation is to remote to the host machine and access the virtual machines through the host connection.
7. If editing the script to use mouse movements, disable "Mouse Integration" for the virtual machine and the remote connection. "Mouse Integration" prevents the macro from controlling the mouse, however it does not necessarily disable mouse clicks.
8. The macro will finish queueing MRNs to be exported from MUSE before the XMLs are actually written to disk. This is because MUSE takes longer to format and write XMLs than it does for the macro to search for MRNs. Simply wait for the exports to finish before moving the extracted XMLs.
9. Sublime Text is the editor chosen because it can cut lines using `Ctrl + X`. Any other editor that supports this capability would also be supported.
10. The macro ends when the cut from Sublime Text is just a Windows style newline. If this feature is not working, the clipboard "empty" detection is a place to start debugging.
11. The macro may break on very large MRN lists when MUSE Editor crashes or the print queue fails for some reason. It is recommended to batch the MRN extraction.

## Organizing XMLs and Removing Duplicates
`1_organize_xml_into_yyyymm.py` moves XML files from a single directory into the appropriate yyyy-mm directory.

`2_remove_xml_duplicates.py` finds and removes exact duplicate XML files, as defined by every bit of two files being identical, determined via SHA-256 hashing. 

## Tensorizing XMLs to HDF5
`python 3_convert_xml_to_hd5.py` extracts data from all XML files and saves as [HDF5 files](https://www.hdfgroup.org). 

This script is called with the `-p` or `--parallel` argument to parallelize conversion across all available CPUs.  

One ECG from one XML is stored as one HDF5 file.  

**This will soon be updated so that all the ECGs for one patient are stored as one HDF5 file.**

## ECG Data Structure
Voltage is saved from XMLs as a dictionary of numpy arrays indexed by leads in the set `("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")`, e.g.:

```
voltage = {'I': array([0, -4, -2, ..., 7]),
          {'II': array([2, -9, 0, ..., 5]),
          ...
          {'V6': array([1, -4, -3, ..., 4]),
```

Every other element extracted from the XML is returned as a string, even if the underlying primitive type is a number (e.g. age). Here are some of the more important elements:

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

## Extracting ECG metadata

`4_extract_metadata_to_csv.py` iterates through every HDF5 file, identifies relevant data (e.g. MRN, diagnostic read, axes, intervals, age, gender, and race), and saves these data in a large CSV file:  

This CSV file will be used to construct a performant, queryable database to identify future cohorts for research projects.

**Metadata extraction will soon change along with new tensorization**

## MUSE Virtual Machine Setup
> Some of these steps will already be complete in the `.ova` image file from `mad3`.  
1. [Download and install VirtualBox](https://www.virtualbox.org/wiki/Downloads).
2. Open File Explorer, click on the navigation bar, type `\\MAD3\MGH-NEURO-CDAC\Projects\partners_ecg\`, and click enter.
3. Log in using your `mgh.harvard.edu` email address and MGH / Partners password.
4. Copy the virtual appliance `muse_mgh.ova` from this directory to your desktop. It will take 4-8 hours.
5. Open VirtualBox, click "Import" (yellow curved arrow icon at the top), select the `.ova` file, and click "next".
6. Modify the "base folder which will host all the virtual machines". You must select a place on your computer with at least 600 GB of storage. **The performance of MUSE Editor within the VM is disk bound, save the VM to a SSD if possible.**  
8. Click Import. It will take 1-2 hours.  
9. After the VM is imported, go back to the Oracle VM VirtualBox Manager home menu. Take a snapshot of the VM and name it `base`. If the VM is corrupted, no need to wait for the VM to reimport, simply restore the snapshot.
10. Configure the VM if desired. Also take a snapshot of the configured VM, name it `configured`. Possible configuration steps:
    1. Enable window resizing and clipboard: Attach an optical drive to the VM. Start the VM and insert "Guest Additions". Follow steps to install "Guest Additions". 
    2. Enable remote desktop: Start the VM and install chrome remote desktop or anydesk to the virtual machine. Use Google Chrome, it should be on the disk image.
