import os
import shutil
from timeit import default_timer as timer
import utils


if __name__ == '__main__':
    # Set path to directory containing XML directories
    fpath_xml = os.path.expanduser('~/partners_ecg/xml')

    # Identify list of date directories within xml/
    fpath_dirs = [f.path for f in os.scandir(fpath_xml) if f.is_dir()]

    # Sort directory names
    fpath_dirs.sort()
   
    # DEBUG CODE
    #fpath_dirs = fpath_dirs[0:5] 

    # Set empty lists
    xmls_and_hashes = []

    start = timer() 

    # Loop through all yyyy-mm/ dirs of XMLs
    for fpath_dir in fpath_dirs:
        # Return list of tuples: (fpath_xml, hash)
        xmls_and_hashes_dir = utils.hash_xml_fname_dir(fpath_dir)
        xmls_and_hashes += xmls_and_hashes_dir
        print('Computed %d hashes for XMLs in %s'
              % (len(xmls_and_hashes_dir), fpath_dir) )

    end = timer()

    print('Hashing %d XML files took %1.2f sec'
          % (len(xmls_and_hashes), end - start) )

    # Sort list of tuples by the hash
    start = timer()
    xmls_and_hashes = sorted(xmls_and_hashes, key = lambda x: x[1])
    end = timer()
    print('Sorting list of (fpath_xml, hash) by hash took %1.2f sec'
          % (end - start) )

    # Find all duplicates from this large list

    # Initialize first hash
    prev_xml = xmls_and_hashes[0][0]
    prev_hash = xmls_and_hashes[0][1]

    dup_count = 0

    start = timer()

    # Loop through all hashes, starting at the second entry
    for xml_and_hash in xmls_and_hashes[1:]:
        
        # If the hash matches the previous, it is a duplicate
        if xml_and_hash[1] == prev_hash:

            # Increment counter
            dup_count += 1

            # Delete duplicate XMLs
            os.remove(xml_and_hash[0])

            ## Move duplicate XML into duplicates directory
            #shutil.move(xml_and_hash[0], fpath_xml_dup)

            print('%s is a duplicate' % xml_and_hash[0])

        # If not, update previous hash
        else:
            prev_xml = xml_and_hash[0]
            prev_hash = xml_and_hash[1]

    end = timer()
    print('Removing %d duplicates / %d ECGs (%1.2f%%) took %1.2f sec'
          % (dup_count,
             len(xmls_and_hashes),
             dup_count / len(xmls_and_hashes) * 100,
             end - start) )
    