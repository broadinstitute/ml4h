# How to extract MUSE ECG data to XMLs

## 1. Open the MUSE Editor
1. Connect to a Virtual Machine with MUSE using chrome remote desktop (https://remotedesktop.google.com/access)
2. Log in to *MuseAdmin* with password *Muse!Admin*.   
3. If your trial of Windows is expired, extend it. Click on the Start menu, type "cmd" or otherwise find the Command Prompt, right-click the icon, and click "Run as Administrator. In the command prompt, type `slmgr /rearm`, hit enter, and follow the prompt to restart Windows 2008 (in the VM). Reconnect via chrome remote desktop.
4. Go to Desktop, open "Services".  
5. Select "MUSE", click on the blue hyperlinked "Start" in "Start the service", then close "services".  
6. Go to Desktop and open the MUSE Editor by clicking on the shortcut to the app.  

## 2. Set up export folder in MUSE Editor
1. Open file explorer and go to `C:\`. Create a new folder (ctrl + shift + n) and name it `sts`.
2. Go to MUSE Editor and go to device setup: System -> Setup (ctrl + shift + p). The MUSE software will close and a new window will open.
3. Add a new folder by clicking the "New" icon in the top bar (between printer and delete icons) and selecting "Folder". Alternatively, go to the top horizontal navigation bar, click "Action" -> "New" -> "Folder". A new window will appear called "Device Properties - Folder".  
4. Fill in the "Device Name" field with the name `00sts`. **Name with 00 so that it shows up at the top of your printer list**.
5. Fill in the "Destination" field with the full path to the folder `C:\sts`. Make sure you capitalize the `C`!
6. Under both "File Extension" and "Output Type", select XML.  
7. Check all three boxes under "Output Options", including "Convert Statement Codes to Text", "Include Measurement Matrix", and "Include Waveforms".  
8. Click "OK".

## 3. Search for ECGs using MRNs in MUSE Editor
1. From the screen in section 2, go to edit/retrieve: System -> Edit/Retrieve (ctrl + shift + e). The MUSE software will close and a new window will open.
2. Enter a MRN in the "Patient ID" field of the Test/Order box in the bottom left hand corner. Click "Search". ![MUSE Editor Test Search Box](https://github.com/mit-ccrg/cardiac-surgery/blob/master/docs/MUSE-test-search-field.png)
3. A list of ECGs should populate in the box at the bottom of the screen. Highlight this entire list (click the first row, hold shift, click the last row). 
4. Once the entire list is selected, find the "Print Test" button in the top menu bar (it looks like a printer). Click the "Print Test" button. ![MUSE Editor Print Test Button](https://github.com/mit-ccrg/cardiac-surgery/blob/master/docs/MUSE-print-test-button.png)
5. A popup screen opens with a list of devices/printers. Select the device you set up in section 2 as the output folder, it will probably be the first item in that list and already highlighted. Set the number of copies to 1, the priority to normal, do not check temporary device, select "Use the default..." for formatting and leave recipient name blank. These should be the default options. Click ok. This should now export the ECG as an XML to the folder from section 2.
6. Repeat steps 2-5 for each MRN.
7. **If you cannot find an MRN, give the MRN to Steven.**

## 4. Move extracted XMLs to Partners DropBox
1. Occasionally, it will be a good idea to "save" by moving all the extracted ECG XMLs to the Partners DropBox. Open Google Chrome on the virtual machine and go to our shared DropBox.
3. Move all the XMLs from `C:\sts` to `[DropBox]\data\muse\xml`.
4. It will be a good idea to move the now transferred XMLs out of `C:\sts` but still keep them on the local disk. Perhaps move them to a folder `C:\backup`? But this way, we ensure we do not create duplicates in MAD3.
