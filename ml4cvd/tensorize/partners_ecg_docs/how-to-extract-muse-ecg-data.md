# How to extract MUSE ECG data to XMLs

Progress on data conversion is tracked at https://docs.google.com/spreadsheets/d/1dXYRk_060l-M4jhG0OLMjhlkma9tPAhQiLJDkXTdwiE/edit#gid=0  

To extract data, you need 800 GB of space on your HDD. The VM is 183 GB as an image, and 600 GB after you open it. The image contains Windows, MUSE software, and all of the MUSE data.

## 1. Set up Windows 2008 VM
> Some of these steps will already be complete in the `.ova` image file from `mad3`.  
1. Download and install VirtualBox from https://www.virtualbox.org/wiki/Downloads.
2. Open File Explorer, click on the navigation bar, type `\\MAD3\MGH-NEURO-CDAC\Projects\MUSE`, and click enter.
3. Log in using your `mgh.harvard.edu` email address and MGH / Partners password.
4. Copy the virtual appliance `muse_mgh.ova` from this directory to your desktop. It will take 4-8 hours.
5. Open VirtualBox, click "Import" (yellow curved arrow icon at the top), select the `.ova` file, and click "next".
6. Modify the "base folder which will host all the virtual machines". You must select a place on your computer with at least 600 GB of storage.   
8. Click Import. It will take 1-2 hours.  
9. After the VM is imported, go back to the Oracle VM VirtualBox Manager home menu. Change boot order; remove optical disk drive, and make HDD the top.
10. **MAKE A SNAPSHOT OF THE VM!** Name it "base". MUSE Software eventually breaks (approx 2 days into lifespan). Revert back to the base snapshot of the image when this happens.

## 2. Open the MUSE editor software
1. Start the VM in VirtualBox. The entire workflow described here is in the VM.
2. MGH: log in to *MuseAdmin* with password *Muse!Admin*.  
    Brigham: log in to *musebkgnd* with password *Muse!Bkgnd*.  
3. If your trial of Windows is expired, extend it. Click on the Start menu, type "cmd" or otherwise find the Command Prompt, right-click the icon, and click "Run as Administrator. In the command prompt, type `slmgr /rearm`, hit enter, and follow the prompt to restart Windows 2008 (in the VM).
4. Go to Desktop, right-click on "services", and select "Run as administrator".  
5. Select "MUSE", click on the blue hyperlinked "Start" in "Start the service", then close "services".  
6. Go to Desktop and open the MUSE Editor by clicking on the shortcut to the app.  

## 3. Set up search parameters to the dates of interest 
* Open file explorer and go to `C:\`. Create a new folder (ctrl + shift + n) and name it the search date, e.g. "2005-03". Before clicking away from naming the folder, select all (ctrl + a) and copy this folder name (ctrl + c).
* Go to MUSE Editor and go to database search: upper-left horizontal navigation bar -> "System" -> "Database Search" (ctrl + shift + d). The MUSE software will close, and a new window will open. 
* In the left vertical navigation bar called "Searches and results", click on "Scheduled searches". Double-click any search to open the "Template search setup".  
* Change the "Report Title". Click and drag the field, paste the name from before (ctrl + v) e.g. "2005-03".  
* Under "Date Field", select "Confirmed Date".  
* Under "Scheduling", select "Run Once Now".  
* Select the appropriate date range (Arrow keys move between month/day/year, tab moves between start and end dates).  
* Click "Ok" (or "Apply" then "Close").  

## 4. Set up the "device folder" 
* Go to device setup: System -> Setup (ctrl + shift + p). The MUSE software will close, and a new window will open.  
* Add a new folder by clicking the "New" icon in the top bar (between printer and delete icons) and selecting "Folder". Alternatively, go to the top horizontal navigation bar, click "Action" -> "New" -> "Folder". A new window will appear called "Device Properties - Folder".  
* Fill in the "Device Name" field with the same name as the folder, e.g. "2005-03". ctrl + A does not work in Muse Software, click and drag to select device name field. Paste the name from before (ctrl + v). Press Tab, it should select all of the next field.  
* Fill in the "Destination" field with the full path to the folder, e.g. `C:\2005-03`. Make sure you capitalize the `C`! If the field is highlighted, type `C:\` and then paste the name from before (ctrl + v).  
* Under both "File Extension" and "Output Type", select XML.  
* Check all three boxes under "Output Options", including "Convert Statement Codes to Text", "Include Measurement Matrix", and "Include Waveforms".  
* Click "OK".
* If repeating these steps for the next time, it is often simpler to edit the same folder from last time by double clicking the entry. This way, the folder will likely appear at the top of the device list and save you a few clicks later.

## 5. Extract MUSE ECG data into the folder 
* Go to the upper-left horizontal navigation bar -> "System" -> "Database Search" (ctrl + shift + D). The MUSE software will close, and a new window will open.  
* Go to the left vertical navigation bar and click on "Search results".
* Double-click on your search, e.g. "2005-03". A new window will pop up.
* Check the box for "Display full result set" if you can. Sometimes this option is greyed out and unselectable.
* Click "Print all tests". A window pops up asking if you "really want to print N records". **If it asks to print 5000 records, double check you've checked "Display full result set".** Click "Yes"; a new window appears.
* In the list of "Available Devices", find and select the one you created, e.g. "2005-03". If you only have one device folder named with numbers, it should be at the top of the list.
* Click "OK". The software will freeze, but MUSE data is being extracted into XML format.
* Switch back to file explorer

> Extracting 30,000 MUSE files to XML takes about ten hours. This is a rate of 50 XMLs per minute.

## 6. Connect to `MAD3` and copy the XML files there
* Select the folder with the extracted XML files. View folder properties (alt + enter) and check that the number of files in the folder is greater than or equal to the number of records found in the search results. More files is okay, they're likely duplicates we can remove. Less files requires rerunning the extraction to ensure no records are missed. If the number of files makes sense for the number of records found for the given date range, move files to mad3.
* Open File Explorer, go to the navigation bar (path), enter `\\mad3\mgh-neuro-cdac\`, and hit enter or return.
* Log in using your MGH email and password.
* Go to Projects -> partners_ecg. The final path is `\\MAD3\MGH-NEURO-CDAC\Projects\partners_ecg`.

> I recommend creating a shortcut to this folder on your desktop or in file explorer sidebar to simplify access.

* Copy the directory you just created (e.g. "2005-03") into the `mgh-raw-xml` directory within the MUSE directory.
* Update the [spreadsheet](https://docs.google.com/spreadsheets/d/1dXYRk_060l-M4jhG0OLMjhlkma9tPAhQiLJDkXTdwiE/edit#gid=368382433) with your progress, and update Erik Reinertsen on Slack.
