#!/bin/python3

# Imports: standard library
import os
import re
import sys
import time
import socket
import logging
import argparse
import datetime
import subprocess
import multiprocessing as mp

q = mp.Queue()
env = os.environ.copy()


def _get_path_to_ecgs() -> str:
    """Check the hostname of the machine and return the appropriate path.
    If there is no match found, this function does not return anything, and
    the script ends up with a non-viable path prefix to HD5 files and will fail."""
    if "anduril" == socket.gethostname():
        path = "/media/4tb1/ecg"
    elif "mithril" == socket.gethostname():
        path = "/data/ecg"
    elif "stultzlab" in socket.gethostname():
        path = "/storage/shared/ecg"
    return os.path.expanduser(path)


def _get_path_to_bootstraps() -> str:
    """Check the hostname of the machine and return the appropriate path.
    If there is no match found, this function does not return anything, and
    the script ends up with a non-viable path prefix to HD5 files and will fail."""
    if "anduril" == socket.gethostname():
        path = "~/dropbox/sts-data/bootstraps"
    elif "mithril" == socket.gethostname():
        path = "~/dropbox/sts-data/bootstraps"
    elif "stultzlab" in socket.gethostname():
        path = "/storage/shared/sts-data-deid/bootstraps"

    path = os.path.expanduser(path)
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a valid path to STS data")
    else:
        return path


def worker(script, bootstrap, gpu):
    env["GPU"] = gpu
    env["BOOTSTRAP"] = bootstrap
    env["PATH_TO_ECGS"] = _get_path_to_ecgs()
    env["PATH_TO_BOOTSTRAPS"] = _get_path_to_bootstraps()

    subprocess.run(
        f"bash {script}".split(),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    q.put(gpu)


def run(args):
    start = time.time()

    for gpu in args.gpus:
        q.put(gpu)

    processes = []
    for script in args.scripts:
        for bootstrap in args.bootstraps:
            gpu = q.get(block=True)
            process = mp.Process(target=worker, args=(script, bootstrap, gpu))
            process.start()
            processes.append(process)
            logging.info(
                f"Dispatched {os.path.basename(script)} with bootstrap {bootstrap} on"
                f" GPU {gpu}",
            )

    for process in processes:
        process.join()

    logging.info(
        f"Dispatched {len(processes)} jobs in {time.time() - start:.0f} seconds",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpus",
        nargs="+",
        required=True,
        help="List of gpu devices to run on, specified by indices or intervals. For example, to use gpu 0, 3, 4, and 5: --gpus 0 3-5",
    )
    parser.add_argument(
        "--bootstraps",
        nargs="+",
        default=["0-9"],
        help="List of bootstraps to run on; same specification as gpus. default: 0-9",
    )
    parser.add_argument(
        "--scripts", nargs="+", required=True, help="list of paths to scripts to run",
    )
    parser.add_argument(
        "--log_file",
        default=f"dispatch-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log",
        help="name of log file to write dispatcher logs to",
    )
    args = parser.parse_args()

    log_formatter = logging.Formatter(
        "%(asctime)s - %(module)s:%(lineno)d - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(args.log_file)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(log_formatter)
    logger.addHandler(stdout_handler)

    interval = re.compile(r"^\d+-\d+$")
    if args.gpus is None or args.bootstraps is None or args.scripts is None:
        raise ValueError(f"Missing arguments")

    def _parse_index_list(idxs, name):
        _idxs = set()
        for idx in idxs:
            if idx.isdigit():
                start = int(idx)
                end = start + 1
            elif interval.match(idx):
                start, end = map(int, idx.split("-"))
                end += 1
            else:
                raise ValueError(f"Invalid {name}: {idx}")
            _idxs = _idxs.union(set(range(start, end)))
        return list(map(str, _idxs))

    args.gpus = _parse_index_list(args.gpus, "gpu")
    args.bootstraps = _parse_index_list(args.bootstraps, "bootstrap")

    for script in args.scripts:
        if not os.path.isfile(script):
            raise ValueError(f"No script found at: {script}")

    return args


if __name__ == "__main__":
    try:
        args = parse_args()
        run(args)
    except Exception as e:
        logging.exception(e)
