import os
import csv
from getpass import getuser
from collections import defaultdict

JOIN_CHAR = '_'


def _clean_label_string(string):
    '''Replace spaces and slashes with JOIN_CHAR,
       and remove parentheses and commas'''
    string = string.replace(' ', JOIN_CHAR)
    string = string.replace('/', JOIN_CHAR)
    string = string.replace('(', '')
    string = string.replace(')', '')
    string = string.replace(',', '')
    return string


def _write_tmap_to_py(py_file, label_maps, channel_maps):
    '''Given label_maps (which associates labels with source phrases)
       and channel_maps (which associates labels with unique sublabels),
       define the tensormaps to associate source phrases with precise labels,
       and write these maps in a python script
    '''
    for label in label_maps:
        cm = '{'

        for i, channel in enumerate(channel_maps[label]):
            cm += f"'{channel}': {i}, "

        # At this point, i = len(channel_maps[label])-1
        # If 'unspecified' is not a label, we need to add and index it
        if 'unspecified' not in channel_maps[label]:
            cm += f"'unspecified': {i+1}"

        cm += '}'

        py_file.write(f"TMAPS['{label}'] = TensorMap('{label}', group='categorical', channel_map={cm}, tensor_from_file=_make_partners_csv_tensors({label_maps[label]})) \n\n")


def _write_partners_ecg_tmap_script(py_file, partners_ecg_label_dir):
    # Iterate through all files in the partners CSV labels folder
    for file in os.listdir(partners_ecg_label_dir):

        # Ignore files that do not end in .csv
        if not file.endswith('.csv'):
            continue

        # Isolate the task name
        task = file.replace('.csv', '').replace('c_', '')

        # Create list of lists;
        # outer list is rows in CSV,
        # inner list is columns; idx 0 is source phrase
        #                        idx 1 is label (level 1 of hierarchy)
        #                        idx 2 is label (level 2 of hierarchy) etc.
        fpath_this_task = os.path.join(partners_ecg_label_dir, file)
        lol = list(csv.reader(open(fpath_this_task), delimiter=','))

        # Associate labels with source phrases in dict of dicts:
        #   keys   - task name and all oot-level labels in hierarchy
        #   values - dicts:
        #       keys   - labels (next level down in hierarchy)
        #       values - list of source phrases that map to a given label
        # Note: because the first key is the task name, keys of dicts in
        # label_map[task] are the remaining keys in label_map itself
        label_maps = defaultdict(dict)

        # Associate labels with unique set of sublabels in dict of sets
        # keys   - every label in hierarchy with children
        # values - set of all child labels within a given label
        channel_maps = defaultdict(set)

        # Iterate through every source phrase in list of lists (label map)
        for row in lol:

            # Initialize prefix as empty list
            prefix = []

            # For the row, iterate through label strings 2:end
            for label_str in row[1:]:

                # If the label string is blank, skip
                if label_str == '':
                    continue

                # Clean label string
                label_str = _clean_label_string(label_str)
               
                # If ??? 
                if len(prefix) == 0:
                    channel_maps[task].add(label_str)

                    # If we already have a list of source phrases for this
                    # task and label string, append the source phrase to it
                    if label_str in label_maps[task]:
                        label_maps[task][label_str].append(row[0])

                    # If no list exists yet for this task and label string,
                    # create a new list and initialize with the source phrase
                    else:
                        label_maps[task][label_str] = [row[0]]
                # ???
                else:
                    #print('===== prefix:', prefix, '=====')
                    
                    # TODO I don't think we need to call join because the label_str
                    # should already be a single string with spaces replaced by
                    # JOIN_CHARs

                    # Join elements in prefix list by JOIN_CHAR
                    prefix_merged = JOIN_CHAR.join(prefix)

                    # ?
                    channel_maps[prefix_merged].add(label_str)

                    if label_str in label_maps[prefix_merged]:
                        label_maps[prefix_merged][label_str].append(row[0])
                    else:
                        label_maps[prefix_merged][label_str] = [row[0]]

                    label_maps[prefix_merged][label_str].append(row[0])

                prefix.append(label_str)

        _write_tmap_to_py(py_file, label_maps, channel_maps)

    breakpoint()
    print('done')


if __name__ == '__main__':
    # Set paths to Dropbox and subdirectory with c_task.csv label maps
    fpath_dropbox = '/home/' + getuser() + '/Dropbox (Partners HealthCare)'
    subdir = 'partners_ecg/partners_ecg_labeling'
    fpath_csv_dir = os.path.join(fpath_dropbox, subdir)
    
    # Set name of python script to create
    script_name = 'tensor_maps_partners_ecg.py'

    # Open the python script, parse the .csv label maps,
    # and create code that creates TensorMaps
    with open(script_name, 'w') as py_file:
        _write_partners_ecg_tmap_script(py_file, fpath_csv_dir)
