import os
import utils
import shutil
import logging
from timeit import default_timer as timer

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--src",
                        default="/data/partners_ecg/xml",
                        help="Path to directory containing organized source XMLs")

    args = parser.parse_args()
    _process_args(args)
    return args


if __name__ == "__main__":

    args = parse_args()
    # Set path to directory containing XML directories
    fpath_xml = args.src

    # Identify list of date directories within xml/
    fpath_dirs = [f.path for f in os.scandir(fpath_xml) if f.is_dir()]
    fpath_dirs.sort()
    xmls_and_hashes = []

    start = timer()

    # Loop through all yyyy-mm/ dirs of XMLs
    for fpath_dir in fpath_dirs:
        # Return list of tuples: (fpath_xml, hash)
        xmls_and_hashes_dir = utils.hash_xml_fname_dir(fpath_dir)
        xmls_and_hashes += xmls_and_hashes_dir
        logging.info(f"Computed {len(xmls_and_hashes_dir)} hashes for XMLs in {fpath_dir}")

    end = timer()
    logging.info(f"Hashing {len(xmls_and_hashes)} XML files took {end-start:.2f} sec")

    # Sort list of tuples by the hash
    start = timer()
    xmls_and_hashes = sorted(xmls_and_hashes, key = lambda x: x[1])
    end = timer()
    logging.info(f"Sorting list of (fpath_xml, hash) by hash took {(end-start):.2f} sec")

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
            logging.info(f"{xml_and_hash[0]} is a duplicate")

        # If not, update previous hash
        else:
            prev_xml = xml_and_hash[0]
            prev_hash = xml_and_hash[1]

    end = timer()
    logging.info(f"Removing {dup_count} duplicates / {len(xmls_and_hashes)} ECGs \
                   ({dup_count / len(xmls_and_hashes) * 100:.2f}%) took {end-start:.2f} sec")
