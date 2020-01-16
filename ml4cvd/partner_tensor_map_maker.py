import os
import csv
from collections import defaultdict

JOIN_CHAR = '_'


def _write_partners_csv(f, partners_csv_folder):
    for file in os.listdir(partners_csv_folder):
        if not file.endswith('.csv'):
            continue
        root_key = file.replace('.csv', '').replace('c_', '')
        lol = list(csv.reader(open(os.path.join(partners_csv_folder, file)), delimiter=','))
        d = defaultdict(dict)
        channel_maps = defaultdict(set)
        for l in lol:
            prefix = []
            for column in l[1:]:
                if column == '':
                    continue
                if len(prefix) == 0:
                    column = column.replace(' ', JOIN_CHAR)
                    channel_maps[root_key].add(column)
                    if column in d[root_key]:
                        d[root_key][column].append(l[0])
                    else:
                        d[root_key][column] = [l[0]]

                else:
                    channel_maps[JOIN_CHAR.join(prefix)].add(column)
                    if column in d[JOIN_CHAR.join(prefix)]:
                        d[JOIN_CHAR.join(prefix)][column].append(l[0])
                    else:
                        d[JOIN_CHAR.join(prefix)][column] = [l[0]]
                    d[JOIN_CHAR.join(prefix)][column].append(l[0])

                prefix.append(column)

        for k in d:
            cm = '{'
            for i, channel in enumerate(channel_maps[k]):
                cm += f"'{channel}': {i}, "
            if 'unspecified' not in channel_maps[k]:
                cm += f"'unspecified': {i+1}"
            cm += '}'
            print(cm)
            print(k)
            f.write(f"TMAPS['{k}'] = TensorMap('{k}', group='categorical', channel_map={cm}, tensor_from_file=_make_partners_csv_tensors({d[k]})) \n\n")


if __name__ == '__main__':
    partners_csv_folder = '/Users/sam/Dropbox/fake_muse/'
    with open('partners_tensor_maps.py', 'w') as f:
        _write_partners_csv(f, partners_csv_folder)