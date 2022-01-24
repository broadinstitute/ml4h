import os
import re
import logging
from typing import Dict, Tuple, Callable

import h5py
import numpy as np

from ml4h.defines import TENSOR_EXT
from ml4h.tensormap.general import get_tensor_at_first_date


def token_dictionary_and_text_from_file(
        text_file: str,
        remove_special_chars: bool = True,
) -> Tuple[str, Dict[str, int]]:
    texts = []
    characters = set()
    with open(text_file) as file:
        for i, line in enumerate(file.readlines()):
            cur_line = _preprocess_sentence(line, remove_special_chars)
            [characters.add(char) for char in cur_line]
            texts.append(cur_line)
            if i % 50000 == 0:
                logging.info(f'Read {i} lines from {text_file}')
    logging.info(f'Total characters: {len(characters)}')
    char2index = dict((c, i) for i, c in enumerate(sorted(list(characters))))
    index2char = dict((i, c) for i, c in enumerate(sorted(list(characters))))
    logging.info(f'char2index:\n\n {char2index}  \n\n\n\n index2char: \n\n {index2char} \n\n\n')
    return ''.join(texts), char2index


def token_dictionary_from_hd5_key(
        tensors: str,
        path_prefix: str,
        name: str,
) -> Dict[str, int]:
    characters = set()
    for i, tp in enumerate(os.listdir(tensors)):
        if os.path.splitext(tp)[-1].lower() != TENSOR_EXT:
            continue
        if i % 400 == 0:
            logging.debug(f'Found {len(characters)} unique tokens in {i} HD5 files at:{tensors}')
        if i > 2000:
            break
        with h5py.File(tensors + tp, 'r') as hd5:
            if name in hd5[path_prefix]:
                characters.update(np.unique(get_tensor_at_first_date(hd5, path_prefix, name)))

    logging.info(f'Total characters from HD5 Tensor {path_prefix} and name {name}: {len(characters)}')
    char2index = dict((c, i) for i, c in enumerate(sorted(list(characters))))
    logging.info(f'char2index:\n {char2index} \n')
    return char2index


def random_text_window_tensor(
    text: str,
    window_size: int,
) -> Callable:
    def text_from_file(tm, _, dependents={}):
        tensor = np.zeros(tm.shape, dtype=np.float32)
        random_index = np.random.randint(window_size, len(text)-window_size)
        for i, c in enumerate(text[random_index:random_index+window_size]):
            tensor[i] = tm.channel_map[c]
        if tm.dependent_map is not None:
            start_next_window = random_index + 1
            dependents[tm.dependent_map] = np.zeros(tm.dependent_map.shape, dtype=np.float32)
            for j, c in enumerate(text[start_next_window:start_next_window + tm.dependent_map.shape[0]]):
                dependents[tm.dependent_map][j] = tm.dependent_map.channel_map[c]
        return tensor
    return text_from_file


def _preprocess_sentence(sentence, remove_special_chars):
    sentence = sentence.strip()
    if remove_special_chars:
        #replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
        sentence = sentence.strip()
    return sentence


def random_array_window_tensors(
    window_shape: Tuple[int],
    shift_axis: int = 0,
) -> Callable:
    def window_as_text_from_file(tm, hd5, dependents={}):
        full_tensor = get_tensor_at_first_date(hd5, tm.path_prefix, tm.name)
        indexes = [np.random.randint(window_shape[i], edge-window_shape[i]) for i, edge in enumerate(full_tensor.shape)]
        random_window = tuple(slice(index-window_shape[i], index) for i, index in enumerate(indexes))
        next_window = tuple(
            slice((index + 1 if i == shift_axis else index)-window_shape[i], index + 1 if i == shift_axis else index) for i, index in enumerate(indexes)
        )
        tensor = full_tensor[random_window].flatten()
        if tm.dependent_map is not None:
            dependents[tm.dependent_map] = np.zeros(tm.dependent_map.shape, dtype=np.float32)
            flat = full_tensor[next_window].flatten()
            for j, c in enumerate(flat):
                dependents[tm.dependent_map][j] = tm.dependent_map.channel_map[c]
        return tensor
    return window_as_text_from_file
