# Partners ECG
Extracting and tensorizing MUSE 12-lead ECGs

## Table of Contents
1. [Extracting ECGs to XML](#extracting-ecgs-to-xml)
2. [Organizing XMLs and Removing Duplicates](#organizing-xmls-and-removing-duplicates)
3. [Tensorizing XMLs to HDF5](#tensorizing-xmls-to-hdf5)
4. [MUSE Virtual Machine Setup](#muse-virtual-machine-setup)

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

### 2. Set up the export folder in MUSE Editor
1. Open file explorer and go to `C:\`. Create a new folder (Ctrl + Shift + N) and name it `export`.
2. Go to MUSE Editor and go to Device Setup: System -> Setup (Ctrl + Shift + P). The MUSE software will close and a new window will open.
3. Add a new folder by clicking the "New" icon in the top bar and selecting "Folder". ![MUSE New Export Folder](images/MUSE-new-export-folder.png)  
Alternatively, from the top horizontal navigation bar: Action -> New -> Folder. A new window will appear called "Device Properties - Folder".
4. Fill in the "Device Name" field with the name `00export`. (`00` helps put the folder at the top of the print list).
5. Fill in the "Destination" field with the full path to the folder `C:\export`. Make sure you capitalize the `C`!
6. Enter `xml` for "File Extension" and select `XML` for "Output Type".  
7. Check all three boxes under "Output Options" including `Convert Statement Codes to Text`, `Include Measurement Matrix`, and `Include Waveforms`.  
8. Click "OK".

### 3. Search for ECGs in MUSE Editor
There are two ways of searching for ECGs: by MRN or by Date Range

#### a. Search by MRN/PatientID
#### b. Search by Date Range

## Organizing XMLs and Removing Duplicates

## Tensorizing XMLs to HDF5

## MUSE Virtual Machine Setup
