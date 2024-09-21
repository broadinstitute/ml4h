import argparse
import logging
import os
import glob
import json

import tensorflow as tf

from echo_defines import category_dictionaries
from data_descriptions.echo import (
    LmdbEchoStudyVideoDataDescription,
    LmdbEchoStudyVideoDataDescriptionBWH,
)
from model_descriptions.echo import DDGenerator, create_movinet_classifier
from model_descriptions.models_view_cls import create_classifier, create_video_encoder
import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO)
tf.get_logger().setLevel(logging.ERROR)


def main(
    batch_size,
    lmdb_folder,
    is_mgh,
    movinet_ckpt_dir,
    n_input_frames,
    output_file,
    pretrained_ckpt_dir,
    skip_modulo,
    study_df,
    n_partitions,
    partition,
    work_ids_from_file,
    view_annotation_tasks,
):
    input_dd_class = (
        LmdbEchoStudyVideoDataDescription
        if is_mgh
        else LmdbEchoStudyVideoDataDescriptionBWH
    )
    input_dd = input_dd_class(
        lmdb_folder,
        "image",
        [],
        n_input_frames,
        skip_modulo,
    )
    
    if work_ids_from_file:
        with open(f'working_ids_p{partition+1}_of_{n_partitions}.json', 'r') as f:
            working_ids_partitions_curr = json.load(f)
    else:
        # Load the IDs
        df_list = []
        for _, row in study_df.drop_duplicates(subset=["study"]).iterrows():
            if is_mgh:
                # study_id = row["study_id"]
                study_id = row["study"]
                if isinstance(lmdb_folder, list):
                    # lmdb_folder_curr = os.path.join(lmdb_folder[0], f"{study_id}.lmdb")
                    file_exists = 0
                    for lmdb_dir in lmdb_folder:
                        if os.path.exists(os.path.join(lmdb_dir, f"{study_id}.lmdb")):
                            lmdb_folder_curr = os.path.join(lmdb_dir, f"{study_id}.lmdb")
                            # print(lmdb_folder_curr)
                            file_exists = 1
                    if file_exists == 0:
                        with open('studies_not_found_in_folders.txt', 'a') as f:
                            f.write(str(study_id)+'\n')
                        print(f'{study_id} folder not found')
                        continue
                else:
                    lmdb_folder_curr = os.path.join(lmdb_folder, f"{study_id}.lmdb")
                # lmdb_folder_curr = glob.glob(os.path.join(lmdb_folder,'*',str(study_id)+'*'))[0]
                try:
                    log_df = pd.read_parquet(
                        os.path.join(lmdb_folder_curr, f"log_{study_id}.pq"))
                except Exception as e:
                    error_msg = f'{study_id}: {e}'
                    with open('log_files_not_found_in_folders.txt', 'a') as f:
                        f.write(str(error_msg)+'\n')
                    print(f'{study_id} log file not found')
                    continue
                    # os.path.join(lmdb_folder, f"{study_id}.lmdb", f"log_{study_id}.pq"),

                log_df = log_df[log_df["stored"]]
                log_df["sample_id"] = log_df["view"].apply(lambda x: f"mrn_{study_id}_{x}")
            else:
                log_df = row['sample_id']
    #             study_id = f"{row['MRN']}_{row['study_id']}"
    #             log_df = pd.read_csv(
    #             glob.glob(os.path.join(lmdb_folder,f"*{row['MRN']}_{row['study_id']}*",f"log_*_{row['study_id']}*.tsv"))[0],
    #             sep="\t")

    #             # log_df = pd.read_csv(
    #             #     os.path.join(lmdb_folder, f"{study_id}.lmdb", f"log_{study_id}.tsv"),
    #             #     sep="\t",
    #             # )
    #             log_df = log_df[log_df["stored"]]
    #             log_df["sample_id"] = log_df["view"].apply(lambda x: f"{row['MRN']}_{row['study_id']}_{x}")
            df_list.append(log_df)
        if is_mgh:
            log_df = pd.concat(df_list)
            working_ids = sorted(log_df["sample_id"].values.tolist())
        else:
            working_ids = sorted(df_list)
        
        n_part = int(np.ceil(len(working_ids)/n_partitions))
        working_ids_partitions = [working_ids[i:i+n_part] for i in range(0, len(working_ids), n_part)]
        
        for part in range(n_partitions):
            with open(f'working_ids_p{part+1}_of_{n_partitions}.json', 'w') as f:
                json.dump(working_ids_partitions[part], f)
        
        print(f'All working ids partition files saved in {os.getcwd()}')
        working_ids_partitions_curr = working_ids_partitions[partition]
        
    
    # n_part = np.ceil(len(working_ids)/n_partitions)
    # working_ids_partitions = [working_ids[i:i+n_part] for i in range(0, len(working_ids), n_part)]
    
    n = int(np.ceil(len(working_ids_partitions_curr)/150))
    working_ids_split = [working_ids_partitions_curr[i:i+n] for i in range(0, len(working_ids_partitions_curr), n)]
    for working_ids_ind, working_ids_cur in enumerate(working_ids_split):
        print(f'Group no. {working_ids_ind} out of {len(working_ids_split)}')
        body_inference_ids = tf.data.Dataset.from_tensor_slices(working_ids_cur).batch(
            batch_size, drop_remainder=False
        )

        io_inference_ds = body_inference_ids.interleave(
            lambda sample_ids: tf.data.Dataset.from_generator(
                DDGenerator(input_dd, None),
                output_signature=(
                    tf.TensorSpec(
                        shape=(None, n_input_frames, 224, 224, 3), dtype=tf.float32
                    ),
                ),
                args=(sample_ids,),
            )
        ).prefetch(2)

        encoder = create_video_encoder(
            path=movinet_ckpt_dir, input_shape=(n_input_frames, 224, 224, 3)
        )
        # encoder = create_video_encoder(
        #     path=movinet_ckpt_dir, input_shape=(n_input_frames, 224, 224, 3)
        # )
        model = create_classifier(
            encoder,
            trainable=False,
            input_shape=(n_input_frames, 224, 224, 3),
            categories={
                f"annotation_{task}": len(category_dictionaries[task])
                for task in view_annotation_tasks
            },
        )

        model.load_weights(pretrained_ckpt_dir)

        model_predictions = {}
        inference_steps = len(working_ids_cur) // batch_size + int(
            (len(working_ids_cur) % batch_size) > 0.5
        )

        predictions = model.predict(io_inference_ds, steps=inference_steps, verbose=1)
        for i, task in enumerate(view_annotation_tasks):
            model_predictions[f"{task}_prediction"] = predictions[i]

        df = pd.DataFrame()
        df["sample_id"] = working_ids_cur[: len(model_predictions[f"{task}_prediction"])]
        for task in view_annotation_tasks:
            # print(model_predictions[f"{task}_prediction"])
            df[f"{task}_prediction"] = np.argmax(
                model_predictions[f"{task}_prediction"], axis=1
            )
            df[f"{task}_prediction_probability"] = np.max(
                model_predictions[f"{task}_prediction"], axis=1
            )

            tmp = model_predictions[f"{task}_prediction"].copy()
            for r_ind, max_ind in enumerate(df[f"{task}_prediction"]):
                tmp[r_ind,max_ind] = -1

            df[f"{task}_prediction_2nd"] = np.argmax(
                tmp, axis=1
            )
            df[f"{task}_prediction_probability_2nd"] = np.max(
                tmp, axis=1
            )
        df.to_csv(output_file+f'_p{partition+1}_of_{n_partitions}_fnum_{working_ids_ind}.tsv', index=False, sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lmdb_folder", action='append', type=str)
    parser.add_argument("--is_mgh", action="store_true")
    parser.add_argument("--movinet_ckpt_dir", type=str)
    parser.add_argument("--n_input_frames", type=int, default=1)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--pretrained_ckpt_dir", type=str)
    parser.add_argument("--skip_modulo", type=int, default=1)
    parser.add_argument("--study_df", type=str)
    parser.add_argument("--n_partitions", type=int)
    parser.add_argument("--partition", type=int)
    parser.add_argument("--work_ids_from_file", action="store_true")
    parser.add_argument(
        "-v",
        "--view_annotation_tasks",
        choices=["doppler", "view", "quality", "canonical"],
        type=str,
        action="append",
    )

    args = parser.parse_args()

    main(
        args.batch_size,
        args.lmdb_folder,
        args.is_mgh,
        args.movinet_ckpt_dir,
        args.n_input_frames,
        args.output_file,
        args.pretrained_ckpt_dir,
        args.skip_modulo,
        # pd.read_parquet(args.study_df),
        pd.read_csv(args.study_df),
        args.n_partitions,
        args.partition,
        args.work_ids_from_file,
        args.view_annotation_tasks,
    )

    
    
    
