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
1. Open File Explorer and go to `C:\`. Create a new folder (Ctrl + Shift + N) and name it `export`.
2. Go to MUSE Editor and go to Device Setup: System -> Setup (Ctrl + Shift + P). The MUSE software will close and a new window will open.
3. Add a new folder by clicking the "New" icon in the top bar and selecting "Folder". ![MUSE New Export Folder](images/MUSE-new-export-folder.png)  
Alternatively, from the top horizontal navigation bar: Action -> New -> Folder. A new window will appear called "Device Properties - Folder".
4. Fill in the "Device Name" field with the name `00export`. (`00` helps put the folder at the top of the print list).
5. Fill in the "Destination" field with the full path to the folder `C:\export`. Make sure you capitalize the `C`!
6. Enter `xml` for "File Extension" and select `XML` for "Output Type".  
7. Check all three boxes under "Output Options" including `Convert Statement Codes to Text`, `Include Measurement Matrix`, and `Include Waveforms`.  
8. Click "OK".

### 3. Search for ECGs in MUSE Editor
There are two ways of searching for ECGs: [by MRN](#a-search-by-mrnpatientid) or [by Date Range](#b-search-by-date-range)

#### a. Search by MRN/PatientID
1. Go to MUSE Editor and go to Edit/Retrieve: System -> Edit/Retrieve (Ctrl + Shift + E). The MUSE software will close and a new window will open.
2. Enter a MRN in the "Patient ID" field of the Test/Order box in the bottom left hand corner. Click "Search". ![MUSE Editor Test Search Box](images/MUSE-test-search-field.png)
3. A list of ECGs should populate in the box at the bottom of the screen. Highlight this entire list (click the first row, hold Shift, click the last row). 
4. Once the entire list is selected, click the "Print Test" button in the top menu bar. ![MUSE Editor Print Test Button](images/MUSE-print-test-button.png)  
A new window will appear called "Select Device and Formatting Options".

#### b. Search by Date Range
1. Go to MUSE Editor and go to Database Search: System -> Database Search (Ctrl + Shift + D). The MUSE software will close and a new window will open.
2. In the left vertical navigation bar called "Searches and results", click on "Scheduled searches". ![MUSE Editor Database Search](images/MUSE-database-search.png)  
Double-click any search to open "Template search setup".  
3. Change the "Report Title" to the date range of interest e.g. `2005-03` for the entire month of March 2005.
4. Under "Date Field", select `Acquisition Date`.  
5. Under "Scheduling", select `Run Once Now`.  
6. Select the appropriate date range (arrow keys move between month/day/year, tab moves between start and end dates).  
7. Click "Ok" (or "Apply" then "Close").
8. In the left vertical navigation bar called "Searches and results", click on "Search results".
9. Wait for the search result to show up. Refresh the view by clicking refresh in the top menu bar. ![MUSE Editor Search Results Refresh](images/MUSE-search-results-refresh.png)
10. Double click your search result. A new window will appear called "Search Results".
11. If there are `> 5000` records, check the box "Display full result set".
12. Click "Print all tests" and then "Yes". A new window will appear called "Select Device and Formatting Options".

### 4. Export ECGs from MUSE Editor
1. From "Select Device and Formatting Options", select the device you set up in section 2 as the output folder, it will probably be the first item in that list and already highlighted.
2. Set "Number of Copies" to `1`.
3. Set "Priority" to `Normal`
4. Uncheck "Temporary Device"
5. Set "Formatting" to `Use the default...`
6. Leave "Recipient Name" blank.
7. Click "OK". This should now export the ECG as XML to the folder from section 2.
8. If exporting `> 100` ECGs, MUSE Editor will likely freeze. This is normal.
9. Wait for ECGs to finish exporting. If the "Date modified" column in File Explorer for the folder shows the folder was last modified `> 1 hour` ago, the ECGs are likely done exporting.
10. Move the XML files to a data store, like MAD3 (\\MAD3\MGH-NEURO-CDAC\Projects\partners_ecg\) or a Partners DropBox (it is easier to download DropBox Desktop on the VM than to upload via the web browser to DropBox).

## Organizing XMLs and Removing Duplicates

## Tensorizing XMLs to HDF5

## MUSE Virtual Machine Setup
