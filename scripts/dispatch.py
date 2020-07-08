#!/bin/python
#
# usage:
# python dispatch.py \
#     --gpus 0-3 \
#     --bootstraps 0-9 \
#     --scripts \
#         train-simple.sh \
#         train-varied.sh \
#         train-deeper.sh

# Imports: standard library
import os
import re
import time
import argparse
import subprocess
import multiprocessing as mp

q = mp.Queue()
env = os.environ.copy()


def worker(script, bootstrap, gpu):
    env["GPU"] = gpu
    env["BOOTSTRAP"] = bootstrap

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
            print(
                f"Dispatched {os.path.basename(script)} with bootstrap {bootstrap} on"
                f" GPU {gpu}",
            )

    for process in processes:
        process.join()

    print(f"Dispatched {len(processes)} jobs in {time.time() - start:.0f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", help="range of gpu devices to run on, e.g. 0-4")
    parser.add_argument(
        "--bootstraps",
        default="0-9",
        help="range of bootstraps to run on, default: 0-9",
    )
    parser.add_argument("--scripts", nargs="+", help="list of paths to scripts to run")
    args = parser.parse_args()

    interval = re.compile(r"^\d+-\d+$")
    if args.gpus is None or args.bootstraps is None or args.scripts is None:
        raise ValueError(f"Missing arguments")
    if not interval.match(args.gpus):
        raise ValueError(f'Invalid range for GPUs: "{args.gpus}"')
    if not interval.match(args.bootstraps):
        raise ValueError(f'Invalid range for Bootstraps: "{args.bootstraps}"')
    for script in args.scripts:
        if not os.path.isfile(script):
            raise ValueError(f"No script found at: {script}")

    args.gpus = [int(n) for n in args.gpus.split("-")]
    args.gpus = [str(n) for n in range(args.gpus[0], args.gpus[1] + 1)]
    args.bootstraps = [int(n) for n in args.bootstraps.split("-")]
    args.bootstraps = [
        str(n) for n in range(args.bootstraps[0], args.bootstraps[1] + 1)
    ]

    run(args)
