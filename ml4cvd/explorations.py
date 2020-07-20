# Imports: standard library
import os
import csv
import copy
import logging
import datetime
import multiprocessing as mp
from functools import reduce
from collections import OrderedDict, defaultdict

# Imports: third party
import h5py
import numpy as np
import pandas as pd

# Imports: first party
from ml4cvd.plots import SUBPLOT_SIZE
from ml4cvd.defines import IMAGE_EXT
from ml4cvd.TensorMap import TensorMap, Interpretation
from ml4cvd.tensor_generators import test_train_valid_tensor_generators

# fmt: off
# need matplotlib -> Agg -> pyplot
import matplotlib                       # isort:skip
matplotlib.use("Agg")                   # isort:skip
from matplotlib import pyplot as plt    # isort:skip
# fmt: on


class ExploreParallelWrapper:
    def __init__(self, tmaps, paths, num_workers, output_folder, run_id):
        self.tmaps = tmaps
        self.paths = paths
        self.num_workers = num_workers
        self.total = len(paths)
        self.output_folder = output_folder
        self.run_id = run_id
        self.chunksize = self.total // num_workers
        self.counter = mp.Value("l", 1)

    def _hd5_to_disk(self, path, gen_name):
        with self.counter.get_lock():
            i = self.counter.value
            if i % 1000 == 0:
                logging.info(f"Parsing {i}/{self.total} ({i/self.total*100:.1f}%) done")
            self.counter.value += 1

        # each worker should write to it's own file
        pid = mp.current_process().pid
        fpath = os.path.join(
            self.output_folder, self.run_id, f"tensors_all_union_{pid}.csv",
        )
        write_header = not os.path.isfile(fpath)

        try:
            with h5py.File(path, "r") as hd5:
                dict_of_tensor_dicts = defaultdict(dict)
                # Iterate through each tmap
                for tm in self.tmaps:
                    shape = tm.shape if tm.shape[0] is not None else tm.shape[1:]
                    try:
                        tensors = tm.tensor_from_file(tm, hd5)
                        if tm.shape[0] is not None:
                            # If not a multi-tensor tensor, wrap in array to loop through
                            tensors = np.array([tensors])
                        for i, tensor in enumerate(tensors):
                            if tensor is None:
                                break

                            error_type = ""
                            try:
                                tensor = tm.postprocess_tensor(
                                    tensor, augment=False, hd5=hd5,
                                )
                                # Append tensor to dict
                                if tm.channel_map:
                                    for cm in tm.channel_map:
                                        dict_of_tensor_dicts[i][
                                            f"{tm.name}_{cm}"
                                        ] = tensor[tm.channel_map[cm]]
                                else:
                                    # If tensor is a scalar, isolate the value in the array;
                                    # otherwise, retain the value as array
                                    if shape[0] == 1:
                                        if type(tensor) == np.ndarray:
                                            tensor = tensor.item()
                                    dict_of_tensor_dicts[i][tm.name] = tensor
                            except (
                                IndexError,
                                KeyError,
                                ValueError,
                                OSError,
                                RuntimeError,
                            ) as e:
                                if tm.channel_map:
                                    for cm in tm.channel_map:
                                        dict_of_tensor_dicts[i][
                                            f"{tm.name}_{cm}"
                                        ] = np.nan
                                else:
                                    dict_of_tensor_dicts[i][tm.name] = np.full(
                                        shape, np.nan,
                                    )[0]
                                error_type = type(e).__name__
                            dict_of_tensor_dicts[i][
                                f"error_type_{tm.name}"
                            ] = error_type

                    except (
                        IndexError,
                        KeyError,
                        ValueError,
                        OSError,
                        RuntimeError,
                    ) as e:
                        # Most likely error came from tensor_from_file and dict_of_tensor_dicts is empty
                        if tm.channel_map:
                            for cm in tm.channel_map:
                                dict_of_tensor_dicts[0][f"{tm.name}_{cm}"] = np.nan
                        else:
                            dict_of_tensor_dicts[0][tm.name] = np.full(shape, np.nan)[0]
                        dict_of_tensor_dicts[0][f"error_type_{tm.name}"] = type(
                            e,
                        ).__name__

                for i in dict_of_tensor_dicts:
                    dict_of_tensor_dicts[i]["fpath"] = path
                    dict_of_tensor_dicts[i]["generator"] = gen_name

                # write tdicts to disk
                if len(dict_of_tensor_dicts) > 0:
                    keys = dict_of_tensor_dicts[0].keys()
                    with open(fpath, "a") as output_file:
                        dict_writer = csv.DictWriter(output_file, keys)
                        if write_header:
                            dict_writer.writeheader()
                        dict_writer.writerows(dict_of_tensor_dicts.values())
        except OSError as e:
            logging.info(f"OSError {e}")

    def mp_worker(self, worker_idx):
        start = worker_idx * self.chunksize
        end = start + self.chunksize
        if worker_idx == self.num_workers - 1:
            end = self.total
        for path, gen in self.paths[start:end]:
            self._hd5_to_disk(path, gen)

    def run(self):
        workers = []
        for i in range(self.num_workers):
            worker = mp.Process(target=self.mp_worker, args=(i,))
            worker.start()
            workers.append(worker)
        for worker in workers:
            worker.join()


def _tensors_to_df(args):
    generators = test_train_valid_tensor_generators(**args.__dict__)
    tmaps = [tm for tm in args.tensor_maps_in]
    paths = [
        (path, gen.name.replace("_worker", ""))
        for gen in generators
        for worker_paths in gen.path_iters
        for path in worker_paths.paths
    ]

    ExploreParallelWrapper(
        tmaps, paths, args.num_workers, args.output_folder, args.id,
    ).run()

    # get columns that should have dtype 'string' instead of dtype 'O'
    str_cols = ["fpath", "generator"]
    for tm in tmaps:
        if tm.interpretation == Interpretation.LANGUAGE:
            str_cols.extend(
                [f"{tm.name}_{cm}" for cm in tm.channel_map]
                if tm.channel_map
                else [tm.name],
            )
        str_cols.append(f"error_type_{tm.name}")
    str_cols = {key: "string" for key in str_cols}

    # Consolidate temporary CSV files into one dataframe
    base = os.path.join(args.output_folder, args.id)
    temp_files = []
    df_list = []
    for name in os.listdir(base):
        if "tensors_all_union_" in name:
            fpath = os.path.join(base, name)
            _df = pd.read_csv(fpath, dtype=str_cols)
            logging.debug(f"Loaded {fpath} into memory")
            df_list.append(_df)
            logging.debug(f"Appended {fpath} to list of dataframes")
            temp_files.append(fpath)
    df = pd.concat(df_list, ignore_index=True)

    logging.info(
        f"Extracted {len(tmaps)} tmaps from {len(df)} tensors across {len(paths)} hd5"
        " files into DataFrame",
    )

    # remove temporary files
    for fpath in temp_files:
        os.remove(fpath)
    logging.debug(f"Deleted {len(temp_files)} temporary files")
    return df


def _modify_tmap_to_return_mean(tmap: TensorMap) -> TensorMap:
    """Modifies tm so it returns it's mean unless previous tensor from file fails"""
    new_tm = copy.deepcopy(tmap)
    new_tm.shape = (1,)
    new_tm.interpretation = Interpretation.CONTINUOUS
    new_tm.channel_map = None

    def tff(_: TensorMap, hd5: h5py.File, dependents=None):
        return tmap.tensor_from_file(tmap, hd5, dependents).mean()

    new_tm.tensor_from_file = tff
    return new_tm


def tmap_requires_modification_for_explore(tm: TensorMap) -> bool:
    """Whether a tmap has to be modified to be used in explore"""
    if tm.is_continuous():
        return tm.shape not in {(1,), (None, 1)}
    if tm.is_categorical():
        if tm.shape[0] is None:
            return tm.axes() > 2
        else:
            return tm.axes() > 1
    if tm.is_language():
        return False
    return True


def explore(args):
    tmaps = [
        _modify_tmap_to_return_mean(tm)
        if tmap_requires_modification_for_explore(tm)
        else tm
        for tm in args.tensor_maps_in
    ]
    args.tensor_maps_in = tmaps
    fpath_prefix = "summary_stats"
    out_ext = "csv"
    out_sep = ","

    # Iterate through tensors, get tmaps, and save to dataframe
    df = _tensors_to_df(args)

    # By default, remove columns with error_type
    if not args.explore_export_errors:
        cols = [c for c in df.columns if not c.startswith("error_type_")]
        df = df[cols]

    # Remove redundant columns for binary labels
    redundant_cms = []
    if Interpretation.CATEGORICAL in [tm.interpretation for tm in tmaps]:
        for tm in [
            tm for tm in tmaps if tm.interpretation is Interpretation.CATEGORICAL
        ]:
            if tm.channel_map:
                for cm in tm.channel_map:
                    if cm.startswith("no_"):
                        df = df.drop(f"{tm.name}_{cm}", 1)
                        redundant_cms.append(cm)
                    elif cm == "female":
                        df = df.drop(f"{tm.name}_{cm}", 1)
                        redundant_cms.append(cm)

    # Save dataframe to CSV
    fpath = os.path.join(args.output_folder, args.id, f"tensors_all_union.{out_ext}")
    df.to_csv(fpath, index=False, sep=out_sep)
    fpath = os.path.join(
        args.output_folder, args.id, f"tensors_all_intersect.{out_ext}",
    )
    df.dropna().to_csv(fpath, index=False, sep=out_sep)
    logging.info(f"Saved dataframe of tensors (union and intersect) to {fpath}")

    # Check if any tmaps are categorical
    if Interpretation.CATEGORICAL in [tm.interpretation for tm in tmaps]:

        # Iterate through 1) df, 2) df without NaN-containing rows (intersect)
        for df_cur, df_str in zip([df, df.dropna()], ["union", "intersect"]):
            for tm in [
                tm for tm in tmaps if tm.interpretation is Interpretation.CATEGORICAL
            ]:
                counts = []
                counts_missing = []
                if tm.channel_map:
                    for cm in tm.channel_map:
                        key = f"{tm.name}_{cm}"
                        if cm not in redundant_cms:
                            counts.append(df_cur[key].sum())
                            counts_missing.append(df_cur[key].isna().sum())
                else:
                    key = tm.name
                    counts.append(df_cur[key].sum())
                    counts_missing.append(df_cur[key].isna().sum())

                # Append list with missing counts
                counts.append(counts_missing[0])

                # Append list with total counts
                counts.append(sum(counts))

                # Create list of row names
                cm_names = [cm for cm in tm.channel_map if cm not in redundant_cms] + [
                    f"missing",
                    f"total",
                ]

                # Transform list into dataframe indexed by channel maps
                df_stats = pd.DataFrame(counts, index=cm_names, columns=["counts"])

                # Add new column: percent of all counts
                df_stats["percent_of_total"] = (
                    df_stats["counts"] / df_stats.loc[f"total"]["counts"] * 100
                )

                # Save parent dataframe to CSV on disk
                fpath = os.path.join(
                    args.output_folder,
                    args.id,
                    f"{fpath_prefix}_{Interpretation.CATEGORICAL}_{tm.name}_{df_str}.csv",
                )
                df_stats = df_stats.round(2)
                df_stats.to_csv(fpath)
                logging.info(
                    f"Saved summary stats of {Interpretation.CATEGORICAL} {tm.name}"
                    f" tmaps to {fpath}",
                )

    # Check if any tmaps are continuous
    if Interpretation.CONTINUOUS in [tm.interpretation for tm in tmaps]:

        # Iterate through 1) df, 2) df without NaN-containing rows (intersect)
        for df_cur, df_str in zip([df, df.dropna()], ["union", "intersect"]):
            df_stats = pd.DataFrame()
            if df_cur.empty:
                logging.info(
                    f"{df_str} of tensors results in empty dataframe. Skipping"
                    f" calculations of {Interpretation.CONTINUOUS} summary statistics",
                )
            else:
                for tm in [
                    tm for tm in tmaps if tm.interpretation is Interpretation.CONTINUOUS
                ]:
                    if tm.channel_map:
                        for cm in tm.channel_map:
                            stats = dict()
                            key = f"{tm.name}_{cm}"
                            stats["min"] = df_cur[key].min()
                            stats["max"] = df_cur[key].max()
                            stats["mean"] = df_cur[key].mean()
                            stats["median"] = df_cur[key].median()
                            mode = df_cur[key].mode()
                            stats["mode"] = mode[0] if len(mode) != 0 else np.nan
                            stats["variance"] = df_cur[key].var()
                            stats["stdev"] = df_cur[key].std()
                            stats["count"] = df_cur[key].count()
                            stats["missing"] = df_cur[key].isna().sum()
                            stats["total"] = len(df_cur[key])
                            stats["missing_percent"] = (
                                stats["missing"] / stats["total"] * 100
                            )
                            df_stats = pd.concat(
                                [
                                    df_stats,
                                    pd.DataFrame([stats], index=[f"{tm.name}_{cm}"]),
                                ],
                            )
                    else:
                        stats = dict()
                        key = tm.name
                        stats["min"] = df_cur[key].min()
                        stats["max"] = df_cur[key].max()
                        stats["mean"] = df_cur[key].mean()
                        stats["median"] = df_cur[key].median()
                        mode = df_cur[key].mode()
                        stats["mode"] = mode[0] if len(mode) != 0 else np.nan
                        stats["variance"] = df_cur[key].var()
                        stats["stdev"] = df_cur[key].std()
                        stats["count"] = df_cur[key].count()
                        stats["missing"] = df_cur[key].isna().sum()
                        stats["total"] = len(df_cur[key])
                        stats["missing_percent"] = (
                            stats["missing"] / stats["total"] * 100
                        )
                        df_stats = pd.concat(
                            [df_stats, pd.DataFrame([stats], index=[key])],
                        )

                # Save parent dataframe to CSV on disk
                fpath = os.path.join(
                    args.output_folder,
                    args.id,
                    f"{fpath_prefix}_{Interpretation.CONTINUOUS}_{df_str}.csv",
                )
                df_stats = df_stats.round(2)
                df_stats.to_csv(fpath)
                logging.info(
                    f"Saved summary stats of {Interpretation.CONTINUOUS} tmaps to"
                    f" {fpath}",
                )

    # Check if any tmaps are language (strings)
    if Interpretation.LANGUAGE in [tm.interpretation for tm in tmaps]:
        for df_cur, df_str in zip([df, df.dropna()], ["union", "intersect"]):
            df_stats = pd.DataFrame()
            if df_cur.empty:
                logging.info(
                    f"{df_str} of tensors results in empty dataframe. Skipping"
                    f" calculations of {Interpretation.LANGUAGE} summary statistics",
                )
            else:
                for tm in [
                    tm for tm in tmaps if tm.interpretation is Interpretation.LANGUAGE
                ]:
                    if tm.channel_map:
                        for cm in tm.channel_map:
                            stats = dict()
                            key = f"{tm.name}_{cm}"
                            stats["count"] = df_cur[key].count()
                            if stats["count"] == 0:
                                stats["count_unique"] = 0
                            else:
                                stats["count_unique"] = len(df_cur[key].value_counts())
                            stats["missing"] = df_cur[key].isna().sum()
                            stats["total"] = len(df_cur[key])
                            stats["missing_percent"] = (
                                stats["missing"] / stats["total"] * 100
                            )
                            df_stats = pd.concat(
                                [
                                    df_stats,
                                    pd.DataFrame([stats], index=[f"{tm.name}_{cm}"]),
                                ],
                            )
                    else:
                        stats = dict()
                        key = tm.name
                        stats["count"] = df_cur[key].count()
                        if stats["count"] == 0:
                            stats["count_unique"] = 0
                        else:
                            stats["count_unique"] = len(df_cur[key].value_counts())
                        stats["missing"] = df_cur[key].isna().sum()
                        stats["total"] = len(df_cur[key])
                        stats["missing_percent"] = (
                            stats["missing"] / stats["total"] * 100
                        )
                        df_stats = pd.concat(
                            [df_stats, pd.DataFrame([stats], index=[tm.name])],
                        )

                # Save parent dataframe to CSV on disk
                fpath = os.path.join(
                    args.output_folder,
                    args.id,
                    f"{fpath_prefix}_{Interpretation.LANGUAGE}_{df_str}.csv",
                )
                df_stats = df_stats.round(2)
                df_stats.to_csv(fpath)
                logging.info(
                    f"Saved summary stats of {Interpretation.LANGUAGE} tmaps to"
                    f" {fpath}",
                )

    for tm in args.tensor_maps_in:
        if tm.interpretation == Interpretation.CONTINUOUS:
            name = tm.name
            plt.figure(figsize=(SUBPLOT_SIZE, SUBPLOT_SIZE * 0.6))
            plt.rcParams.update({"font.size": 14})
            plt.xlabel(name)
            plt.ylabel("Count")
            figure_path = os.path.join(
                args.output_folder, args.id, f"{name}_histogram{IMAGE_EXT}",
            )
            plt.hist(df[name].dropna(), 50, rwidth=0.9)
            plt.savefig(figure_path)
            plt.close()
            logging.info(f"Saved {name} histogram plot at: {figure_path}")


def continuous_explore_header(tm: TensorMap) -> str:
    return tm.name


def categorical_explore_header(tm: TensorMap, channel: str) -> str:
    return f"{tm.name}_{channel}"


def cross_reference(args):
    """Cross reference a source cohort with a reference cohort."""
    cohort_counts = OrderedDict()

    src_path = args.tensors_source
    src_name = args.tensors_name
    src_join = args.join_tensors
    src_time = args.time_tensor
    ref_path = args.reference_tensors
    ref_name = args.reference_name
    ref_join = args.reference_join_tensors
    ref_start = args.reference_start_time_tensor
    ref_end = args.reference_end_time_tensor
    ref_labels = args.reference_labels
    number_in_window = args.number_per_window
    order_in_window = args.order_in_window
    window_names = args.window_name
    match_exact_window = order_in_window is not None
    match_min_window = not match_exact_window
    match_any_window = args.match_any_window
    match_every_window = not match_any_window

    # parse options
    src_cols = list(src_join)
    ref_cols = list(ref_join)
    if ref_labels is not None:
        ref_cols.extend(ref_labels)

    def _cols_from_time_windows(time_windows):
        return {
            time_point[0] for time_window in time_windows for time_point in time_window
        }

    use_time = not any(arg is None for arg in [src_time, ref_start, ref_end])
    if use_time:
        if len(ref_start) != len(ref_end):
            raise ValueError(
                f"Invalid time windows, got {len(ref_start)} starts and {len(ref_end)}"
                " ends",
            )

        if order_in_window is None:
            # if not matching exactly N in time window, order_in_window is None
            # make array of blanks so zip doesnt break later
            order_in_window = [""] * len(ref_start)
        elif len(order_in_window) != len(ref_start):
            raise ValueError(
                f"Ambiguous time selection in time windows, got {len(order_in_window)}"
                f" order_in_window for {len(ref_start)} windows",
            )

        if window_names is None:
            window_names = [str(i) for i in range(len(ref_start))]
        elif len(window_names) != len(ref_start):
            raise ValueError(
                f"Ambiguous time window names, got {len(window_names)} names for"
                f" {len(ref_start)} windows",
            )

        # get time columns and ensure time windows are defined
        src_cols.append(src_time)

        # ref start and end are lists of lists, defining time windows
        time_windows = list(zip(ref_start, ref_end))

        # each time window is defined by a tuples of two lists,
        # where the first list of each tuple defines the start point of the time window
        # and the second list of each tuple defines the end point of the time window
        for start, end in time_windows:
            # each start/end point list is two elements,
            # where the first element in the list is the name of the time tensor
            # and the second element is the offset to the value of the time tensor

            # add day offset of 0 for time points without explicit offset
            [
                time_point.append(0)
                for time_point in [start, end]
                if len(time_point) == 1
            ]

            # parse day offset as int
            start[1] = int(start[1])
            end[1] = int(end[1])

        # add unique column names to ref_cols
        ref_cols.extend(_cols_from_time_windows(time_windows))

    # load data into dataframes
    def _load_data(name, path, cols):
        if os.path.isdir(path):
            logging.debug(f"Assuming {name} is directory of hd5 at {path}")
            # Imports: first party
            from ml4cvd.arguments import _get_tmap

            args.tensor_maps_in = [_get_tmap(it, cols) for it in cols]
            df = _tensors_to_df(args)[cols]
        else:
            logging.debug(f"Assuming {name} is a csv at {path}")
            df = pd.read_csv(path, usecols=cols, low_memory=False)
        return df

    src_df = _load_data(src_name, src_path, src_cols)
    logging.info(f"Loaded {src_name} into dataframe")
    ref_df = _load_data(ref_name, ref_path, ref_cols)
    logging.info(f"Loaded {ref_name} into dataframe")

    # cleanup time col
    if use_time:
        src_df[src_time] = pd.to_datetime(
            src_df[src_time], errors="coerce", infer_datetime_format=True,
        )
        src_df.dropna(subset=[src_time], inplace=True)

        for ref_time in _cols_from_time_windows(time_windows):
            ref_df[ref_time] = pd.to_datetime(
                ref_df[ref_time], errors="coerce", infer_datetime_format=True,
            )
        ref_df.dropna(subset=_cols_from_time_windows(time_windows), inplace=True)

        def _add_offset_time(ref_time):
            offset = ref_time[1]
            ref_time = ref_time[0]
            if offset == 0:
                return ref_time
            ref_time_col = f"{ref_time}_{offset:+}_days"
            if ref_time_col not in ref_df:
                ref_df[ref_time_col] = ref_df[ref_time].apply(
                    lambda x: x + datetime.timedelta(days=offset),
                )
                ref_cols.append(ref_time_col)
            return ref_time_col

        # convert time windows to tuples of cleaned and parsed column names
        time_windows = [
            (_add_offset_time(start), _add_offset_time(end))
            for start, end in time_windows
        ]
    logging.info("Cleaned data columns and removed rows that could not be parsed")

    # drop duplicates based on cols
    src_df.drop_duplicates(subset=src_cols, inplace=True)
    ref_df.drop_duplicates(subset=ref_cols, inplace=True)
    logging.info("Removed duplicates from dataframes, based on join, time, and label")

    cohort_counts[f"{src_name} (total)"] = len(src_df)
    cohort_counts[f'{src_name} (unique {" + ".join(src_join)})'] = len(
        src_df.drop_duplicates(subset=src_join),
    )
    cohort_counts[f"{ref_name} (total)"] = len(ref_df)
    cohort_counts[f'{ref_name} (unique {" + ".join(ref_join)})'] = len(
        ref_df.drop_duplicates(subset=ref_join),
    )

    # merging on join columns duplicates rows in source if there are duplicate join values in both source and reference
    # this is fine, each row in reference needs all associated rows in source
    cross_df = src_df.merge(
        ref_df, how="inner", left_on=src_join, right_on=ref_join,
    ).sort_values(src_cols)
    logging.info("Cross referenced based on join tensors")

    cohort_counts[f'{src_name} in {ref_name} (unique {" + ".join(src_cols)})'] = len(
        cross_df.drop_duplicates(subset=src_cols),
    )
    cohort_counts[f'{src_name} in {ref_name} (unique {" + ".join(src_join)})'] = len(
        cross_df.drop_duplicates(subset=src_join),
    )
    cohort_counts[f"{ref_name} in {src_name} (unique joins + times + labels)"] = len(
        cross_df.drop_duplicates(subset=ref_cols),
    )
    cohort_counts[f'{ref_name} in {src_name} (unique {" + ".join(ref_join)})'] = len(
        cross_df.drop_duplicates(subset=ref_join),
    )

    # dump results and report label distribution
    def _report_cross_reference(df, title):
        title = title.replace(" ", "_")
        if ref_labels is not None:
            series = (
                df[ref_labels]
                .astype(str)
                .apply(lambda x: "<>".join(x), axis=1, raw=True)
            )
            label_values, counts = np.unique(series, return_counts=True)
            label_values = np.array(
                list(map(lambda val: val.split("<>"), label_values)),
            )
            label_values = np.append(
                label_values, [["Total"] * len(ref_labels)], axis=0,
            )
            total = sum(counts)
            counts = np.append(counts, [total])
            fracs = list(map(lambda f: f"{f:0.5f}", counts / total))

            res = pd.DataFrame(data=label_values, columns=ref_labels)
            res["count"] = counts
            res["fraction total"] = fracs

            # save label counts to csv
            fpath = os.path.join(
                args.output_folder, args.id, f"label_counts_{title}.csv",
            )
            res.to_csv(fpath, index=False)
            logging.info(f"Saved distribution of labels in cross reference to {fpath}")

        # save cross reference to csv
        fpath = os.path.join(args.output_folder, args.id, f"list_{title}.csv")
        df.set_index(src_join, drop=True).to_csv(fpath)
        logging.info(f"Saved cross reference to {fpath}")

    if use_time:
        # count rows across time windows
        def _count_time_windows(dfs, title, exact_or_min):
            if type(dfs) is list:
                # Number of pre-op (surgdt -180 days; surgdt) ECG from patients with 1+ ECG in all windows
                # Number of distinct pre-op (surgdt -180 days; surgdt) ECG from patients with 1+ ECG in all windows
                # Number of distinct pre-op (surgdt -180 days; surgdt) ecg_patientid_clean from patients with 1+ ECG in all windows

                # Number of newest pre-op (surgdt -180 days; surgdt) ECG from patients with 1 ECG in all windows
                # Number of distinct newest pre-op (surgdt -180 days; surgdt) ECG from patients with 1 ECG in all windows
                # Number of distinct newest pre-op (surgdt -180 days; surgdt) ecg_patientid_clean from patients with 1 ECG in all windows
                for df, window_name, order, (start, end) in zip(
                    dfs, window_names, order_in_window, time_windows,
                ):
                    order = f"{order} " if exact_or_min == "exactly" else ""
                    start = start.replace("_", " ")
                    end = end.replace("_", " ")
                    cohort_counts[
                        f"Number of {order}{window_name} ({start}; {end}) {src_name}"
                        f" from patients with {title}"
                    ] = len(df)
                    cohort_counts[
                        f"Number of distinct {order}{window_name} ({start}; {end})"
                        f" {src_name} from patients with {title}"
                    ] = len(df.drop_duplicates(subset=src_cols))
                    cohort_counts[
                        f"Number of distinct {order}{window_name} ({start}; {end})"
                        f' {" + ".join(src_join)} from patients with {title}'
                    ] = len(df.drop_duplicates(subset=src_join))
            else:
                # Number of ECGs from patients with 1+ ECG in all windows
                # Number of distinct ECGs from patients with 1+ ECG in all windows
                # Number of distinct ecg_patientid_clean from patients with 1+ ECG in all windows
                df = dfs
                cohort_counts[f"Number of {src_name} from patients with {title}"] = len(
                    df,
                )
                cohort_counts[
                    f"Number of distinct {src_name} from patients with {title}"
                ] = len(df.drop_duplicates(subset=src_cols))
                cohort_counts[
                    f'Number of distinct {" + ".join(src_join)} from patients with'
                    f" {title}"
                ] = len(df.drop_duplicates(subset=src_join))

        # aggregate all time windows back into one dataframe with indicator for time window index
        def _aggregate_time_windows(time_window_dfs, window_names):
            for df, window_name in zip(time_window_dfs, window_names):
                if "time_window" not in df:
                    df["time_window"] = window_name
            aggregated_df = pd.concat(time_window_dfs, ignore_index=True).sort_values(
                by=src_cols + ["time_window"], ignore_index=True,
            )
            return aggregated_df

        # get only occurrences for join_tensors that appear in every time window
        def _intersect_time_windows(time_window_dfs):
            # find the intersection of join_tensors that appear in all time_window_dfs
            join_tensor_intersect = reduce(
                lambda a, b: a.merge(b),
                [
                    pd.DataFrame(df[src_join].drop_duplicates())
                    for df in time_window_dfs
                ],
            )

            # filter time_window_dfs to only the rows that have join_tensors across all time windows
            time_window_dfs_intersect = [
                df.merge(join_tensor_intersect) for df in time_window_dfs
            ]
            return time_window_dfs_intersect

        # 1. get data with at least N (default 1) occurrences in all time windows
        # 2. within each time window, get only data for join_tensors that have N rows in the time window
        # 3. across all time windows, get only data for join_tensors that have data in all time windows

        # get df for each time window
        dfs_min_in_any_time_window = [
            cross_df[
                (cross_df[start] < cross_df[src_time])
                & (cross_df[src_time] < cross_df[end])
            ]
            for start, end in time_windows
        ]

        # get at least N occurrences in any time window
        dfs_min_in_any_time_window = [
            df.groupby(src_join + [start, end]).filter(
                lambda g: len(g) >= number_in_window,
            )
            for df, (start, end) in zip(dfs_min_in_any_time_window, time_windows)
        ]
        if match_min_window and match_any_window:
            min_in_any_time_window = _aggregate_time_windows(
                dfs_min_in_any_time_window, window_names,
            )
            logging.info(
                f"Cross referenced so unique event occurs {number_in_window}+ times in"
                " any time window",
            )
            title = f"{number_in_window}+ in any window"
            _report_cross_reference(min_in_any_time_window, title)
            _count_time_windows(dfs_min_in_any_time_window, title, "at least")
            if len(dfs_min_in_any_time_window) > 1:
                _count_time_windows(min_in_any_time_window, title, "at least")

        # get at least N occurrences in every time window
        if match_min_window and match_every_window:
            dfs_min_in_every_time_window = _intersect_time_windows(
                dfs_min_in_any_time_window,
            )
            min_in_every_time_window = _aggregate_time_windows(
                dfs_min_in_every_time_window, window_names,
            )
            logging.info(
                f"Cross referenced so unique event occurs {number_in_window}+ times in"
                " all time windows",
            )
            title = f"{number_in_window}+ in all windows"
            _report_cross_reference(min_in_every_time_window, title)
            _count_time_windows(dfs_min_in_every_time_window, title, "at least")
            if len(dfs_min_in_every_time_window) > 1:
                _count_time_windows(min_in_every_time_window, title, "at least")

        # get exactly N occurrences, select based on ordering
        def _get_occurrences(df, order, start, end):
            if order == "newest":
                df = df.groupby(src_join + [start, end]).tail(number_in_window)
            elif order == "oldest":
                df = df.groupby(src_join + [start, end]).head(number_in_window)
            elif order == "random":
                df = df.groupby(src_join + [start, end]).apply(
                    lambda g: g.sample(number_in_window),
                )
            else:
                raise NotImplementedError(
                    f"Ordering for which rows to use in time window unknown: '{order}'",
                )
            return df.reset_index(drop=True)

        # get exactly N occurrences in any time window
        if match_exact_window:
            dfs_exact_in_any_time_window = [
                _get_occurrences(df, order, start, end)
                for df, order, (start, end) in zip(
                    dfs_min_in_any_time_window, order_in_window, time_windows,
                )
            ]
        if match_exact_window and match_any_window:
            exact_in_any_time_window = _aggregate_time_windows(
                dfs_exact_in_any_time_window, window_names,
            )
            logging.info(
                f"Cross referenced so unique event occurs exactly {number_in_window}"
                " times in any time window",
            )
            title = f"{number_in_window} in any window"
            _report_cross_reference(exact_in_any_time_window, title)
            _count_time_windows(dfs_exact_in_any_time_window, title, "exactly")
            if len(dfs_exact_in_any_time_window) > 1:
                _count_time_windows(exact_in_any_time_window, title, "exactly")

        # get exactly N occurrences in every time window
        if match_exact_window and match_every_window:
            dfs_exact_in_every_time_window = _intersect_time_windows(
                dfs_exact_in_any_time_window,
            )
            exact_in_every_time_window = _aggregate_time_windows(
                dfs_exact_in_every_time_window, window_names,
            )
            logging.info(
                f"Cross referenced so unique event occurs exactly {number_in_window}"
                " times in all time windows",
            )
            title = f"{number_in_window} in all windows"
            _report_cross_reference(exact_in_every_time_window, title)
            _count_time_windows(dfs_exact_in_every_time_window, title, "exactly")
            if len(dfs_exact_in_every_time_window) > 1:
                _count_time_windows(exact_in_every_time_window, title, "exactly")
    else:
        _report_cross_reference(cross_df, f"all {src_name} in {ref_name}")

    # report counts
    fpath = os.path.join(args.output_folder, args.id, "summary_cohort_counts.csv")
    pd.DataFrame.from_dict(
        cohort_counts, orient="index", columns=["count"],
    ).rename_axis("description").to_csv(fpath)
    logging.info(f"Saved cohort counts to {fpath}")
