import xmltodict
import os
import re
import shutil

# Initialize counters
import utils


if __name__ == '__main__':
    num_bad_encodings = 0

    # Define accepted XML encodings
    valid_encodings = {'UTF-8', 'UTF-16', 'ISO-8859-1'}

    # Set paths manually for debugging
    # fpath_xml_source = os.path.expanduser('~/Desktop')
    fpath_xml_bad = os.path.expanduser('~/muse_data/mgh/bad-xml')
    fpath_xml_final = os.path.expanduser('~/muse_data/mgh/xml')

    #fpath_xml_source = os.path.expanduser('~/muse_data/mgh/xml')

    # Ask user to input path to source XMLs
    fpath_xml_source = os.path.expanduser(input("Path to dir with source XMLs? "))

    # Catch input error: blank input
    if not fpath_xml_source:
        raise NameError('Please enter a valid directory to parse.')

    # Catch input error: path does not exist
    else:
        if not os.path.exists(fpath_xml_source):
            raise NameError('Please enter a valid directory to parse.')

    # Catch input error: blank input
    if not fpath_xml_bad:
        raise NameError('Please enter a valid directory.')
    # Catch input error: path does not exist
    else:
        if not os.path.exists(fpath_xml_bad):
            raise NameError('Please enter a valid directory.')

    # Catch input error: blank input
    if not fpath_xml_final:
        raise NameError('Please enter a valid directory.')
    # Catch input error: path does not exist
    else:
        if not os.path.exists(fpath_xml_final):
            raise NameError('Please enter a valid directory.')

    # Debug code
    print('Path to source XMLs is: ' + fpath_xml_source)
    print('Path to bad XMLs is: ' + fpath_xml_bad)

    # Loop through XML files in the directory
    for filename in os.listdir(fpath_xml_source):

        # Skip non-XML files
        if not filename.endswith('.xml'):
            continue

        print('\n')

        pathToFile = os.path.join(fpath_xml_source, filename)

        print('Parsing XML at:' + pathToFile)

        # Declare logical flag to indicate valid XML
        valid_xml = True

        # Open XML as a Python Dictionary
        with open(pathToFile) as fd:

            # Parse first line of XML to assess encoding;
            # if invalid, replace with valid encoding
            xml_as_string = fd.read()

            # Regex to find 'encoding="STUFF-HERE"' within
            # <?xml version="1.0" encoding="STUFF-HERE"?>''
            # by returning everything after 'encoding="' and before ""
            # re.search(r'(?<=encoding\=\")(.*?)(?=\"\?)', xml_as_string).group(0)

            # 1. Look behind positive (?<=B)A finds A preceded by B
            #    Here, our encoding value is left-bound by 'encoding="'
            # 2. Return one or more characters via reluctant (lazy) match
            # 3. Look ahead postive A(?=B) finds expression A where B follows.
            #    Here, our encoding value is right-bound by '"?>'

            verbosePattern = re.compile("""
                (?<=encoding\=\") 
                (.*?) 
                (?=\"\?\>)
                """, re.VERBOSE)

            # Extract XML encoding from first line of imported XML
            xml_encoding = re.search(verbosePattern, xml_as_string)

            if xml_encoding is None:
                valid_xml = False
            else:
                xml_encoding = xml_encoding.group(0)

            # If xml_encoding is not among the accepted XML encodings, fix it
            if xml_encoding not in valid_encodings or not valid_xml:

                print('Bad XML encoding found: \'', xml_encoding,
                      '\'. Replacing with \'ISO-8859-1\'.', sep='')

                # Replace the bad encoding in xml_as_string with ISO-8859-1
                xml_as_string = re.sub(verbosePattern, 'ISO-8859-1', xml_as_string,
                                       count=1)

                # Increment counter to track bad encodings
                num_bad_encodings += 1

                # Overwrite XML file with corrected encoding
                f = open(pathToFile, "w")
                f.write(xml_as_string)
                f.close()

            try:
                # Parse XMl-as-string into a dict
                doc = xmltodict.parse(xml_as_string)

                # Isolate acquisition date of test and save as mm-dd-yyyy format
                ecg_date = doc['RestingECG']['TestDemographics']['AcquisitionDate']

                print('Acquisition date found! ' + ecg_date)

                # Check if the date
                # 1) has ten digits
                # 2) has a dash at index 2
                # 3) has a dash at index 5
                date_check = [len(ecg_date) == 10,
                              ecg_date[2] == '-',
                              ecg_date[5] == '-']

                # If does not pass date check, inform user XML has bad date format
                if not all(date_check):
                    valid_xml = False

                # If passes date check
                else:

                    # Define full path to new directory in yyyy-mm format
                    yyyymm = utils.format_date(ecg_date,
                                                    day_flag=False)
                    fpath_xml_final_yyyymm = os.path.join(fpath_xml_final, yyyymm)

                    # If directory does not exist, create it
                    if os.path.exists(fpath_xml_final_yyyymm):
                        print('Valid yyyy-mm directory exists: '
                              + fpath_xml_final_yyyymm)
                    else:
                        print('No valid yyyy-mm directory exists. Creating: '
                              + fpath_xml_final_yyyymm)
                        os.makedirs(fpath_xml_final_yyyymm)
            
            # If there is any parsing error, set flag to False
            except xmltodict.expat.ExpatError:
                valid_xml = False

        # If the XML is not valid, set the final path to bad xml path
        if not valid_xml:
            fpath_xml_final_yyyymm = fpath_xml_bad
            print('Missing or invalid acquisition date, or other XML error.')

        # If new directory does not exist, create it
        if not os.path.exists(fpath_xml_final_yyyymm):
            os.makedirs(fpath_xml_final_yyyymm)

        # Copy XML file into new directory
        fpath_xml_newdir = fpath_xml_final_yyyymm + '/' + filename
        shutil.move(pathToFile, fpath_xml_newdir)
        print('XML moved to: ' + fpath_xml_final_yyyymm)

