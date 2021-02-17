# ML4H is released under the following BSD 3-Clause License:
#
# Copyright (c) 2020, Broad Institute, Inc. and The General Hospital Corporation.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name Broad Institute, Inc. or The General Hospital Corporation
#   nor the names of its contributors may be used to endorse or promote products
#   derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import time
import json
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple
import os
import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from fastparquet import ParquetFile
import matplotlib.pyplot as plt
import cv2
import blosc
from ingest_mri import compress_and_store
from two_d_projection import pad_center, build_z_slices


def uncompress(t):
    return np.frombuffer(blosc.decompress(t[()]), dtype=np.uint16).reshape(
        t.attrs["shape"]
    )


def pad_center(img, shape):
    border_v = 0
    border_h = 0
    if (shape[0] / shape[1]) >= (img.shape[0] / img.shape[1]):
        border_v = int((((shape[0] / shape[1]) * img.shape[1]) - img.shape[0]) / 2)
    else:
        border_h = int((((shape[1] / shape[0]) * img.shape[0]) - img.shape[1]) / 2)
    img = cv2.copyMakeBorder(
        img, border_v, border_v, border_h, border_h, cv2.BORDER_CONSTANT, 0
    )
    img = cv2.resize(img, (shape[1], shape[0]))
    return img


def autosegment(img):
    img = (img / img.max() * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    _, markers = cv2.connectedComponents(thresh)
    marker_area = [np.sum(markers == m) for m in range(np.max(markers) + 1) if m != 0]
    marker_area_rank = np.argsort(marker_area)[::-1]
    top2 = np.array(marker_area)[marker_area_rank[:2]]
    #
    countour_length = 0
    is_legs = False
    if ((top2 / top2.max()).min() >= 0.25) and (len(top2) == 2):
        is_legs = True
        thresh = (
            (markers == (marker_area_rank + 1)[0])
            | (markers == (marker_area_rank + 1)[1])
        ).astype(np.uint8)
    else:
        largest_component = np.argmax(marker_area) + 1
        thresh = (markers == largest_component).astype(np.uint8)
    #
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k, iterations=1)
    contour, _ = cv2.findContours(closing, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(closing, [cnt], 0, 255, -1)
    #
    # Recapture surface area
    contour, _ = cv2.findContours(closing, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        countour_length += cv2.arcLength(cnt, True)
    #
    return closing, is_legs, countour_length


def autosegment2(meta: str, file: str, destination: str, instance: int = 2):
    meta = fp.ParquetFile(meta).to_pandas()

    with h5py.File(file, "r") as f:
        data = {
            int(name): uncompress(f[f"instance/{instance}/series/{name}"])
            for name in f[f"instance/{instance}/series"]
        }

    z_pos = meta.groupby("series_number")["image_position_z"].agg(["min", "max"])
    slices = build_z_slices(
        [data[i].shape[-1] for i in range(1, 25, 4)],
        [z_pos.loc[i] for i in range(1, 25, 4)],
    )

    total = []
    scaling_factors = [3.0, 4.5, 4.5, 4.5, 3.5, 4.0]
    for i, s in zip(range(1, 25, 4), scaling_factors):
        d = data[i][..., slices[i // 4].start : slices[i // 4].stop]
        d = zoom(d, (1.0, 1.0, s / 3.0))
        d = pad_center(d, (174, 224, d.shape[-1]))
        total.append(d)
        print(d.shape)

    dat = np.concatenate(total, axis=-1)

    xmm = 2.232142925262451
    ymm = 2.232142925262451
    zmm = 3.0

    # legs_observed = False
    stack = []
    leg_flag = []
    cubic_mm = []
    surface_area = []

    for i in range(dat.shape[-1]):
        closing, is_legs, clen = autosegment(dat[..., i])
        stack.append(closing)
        leg_flag.append(is_legs)
        cubic_mm.append((closing == 255).sum() * (xmm * ymm * zmm))
        surface_area.append(clen)

    #
    stack = np.array(stack).astype(np.uint16)

    # Dataframe of volumes
    p = pd.DataFrame(
        {
            "x": range(len(cubic_mm)),
            "volume": cubic_mm,
            "surface_area": surface_area,
            "is_leg": leg_flag,
        }
    )
    p["volume"] /= 1e6
    p["surface_area"] *= xmm * zmm  # pixel length * mm/pixel * depth of stack in mm
    p["surface_area"] /= 1e6
    p["x_rev"] = p["x"][::-1].values
    p = p.loc[p.index.values[::-1]]
    print(np.all(p.surface_area < p.volume))

    output_name = str(meta.ukbid.iloc[0])
    p.to_parquet(os.path.join(destination, f"{output_name + '.pq'}"))

    with h5py.File(os.path.join(destination, f"{output_name + '.h5'}"), "a") as f:
        compress_and_store(f, stack, f"/instance/{instance}/series/{s}/surface")
        compress_and_store(f, stack, f"/instance/{instance}/series/{s}/axial_stack")

    return True


def _build_projection_hd5s(df, destination: str):
    errors = {}
    name = os.getpid()
    print(f"Starting process {name} with {len(df)} files")
    for i in range(len(df)):
        try:
            autosegment2(df.meta.iloc[i], df.file.iloc[i], destination)
        except Exception as e:
            errors[df.file.iloc[i]] = str(e)
        if len(df) % max(i // 10, 1) == 0:
            print(f"{name}: {(i + 1) / len(df):.2%} done")
    return errors


def multiprocess_project(
    df,
    destination: str,
):
    os.makedirs(destination, exist_ok=True)
    split_files = np.array_split(df, cpu_count())
    print(f"Beginning autosegmentation of {len(df)} samples.")
    start = time.time()
    errors = {}
    with Pool(cpu_count()) as pool:
        results = [
            pool.apply_async(_build_projection_hd5s, (split, destination))
            for split in split_files
        ]
        for result in results:
            errors.update(result.get())
    delta = time.time() - start
    print(f"Projections took {delta:.1f} seconds at {delta / len(df):.1f} s/file")
    with open(os.path.join(destination, "errors.json"), "w") as f:
        json.dump(errors, f)
    return errors
