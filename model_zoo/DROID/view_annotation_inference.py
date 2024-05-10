import argparse
import logging

import tensorflow as tf

from echo_defines import category_dictionaries
from data_descriptions.echo import LmdbEchoStudyVideoDataDescriptionBWH
from model_descriptions.echo import DDGenerator, create_movinet_classifier
from model_descriptions.models import create_classifier, create_video_encoder
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
    view_annotation_tasks,
):
    input_dd = LmdbEchoStudyVideoDataDescriptionBWH(
        lmdb_folder,
        "image",
        [],
        n_input_frames,
        skip_modulo,
    )

    # Load the IDs
    df_list = []
    for _, row in study_df.iterrows():
        if is_mgh:
            study_id = row["study_id"]
            log_df = pd.read_parquet(
                f"work/data/{lmdb_folder}/{study_id}.lmdb/log_{study_id}.pq",
            )
            log_df = log_df[log_df["stored"]]
            log_df["sample_id"] = log_df["view"]
        else:
            study_id = f"{row['MRN']}_{row['study_id']}"
            log_df = pd.read_csv(
                f"work/data/{lmdb_folder}/{study_id}.lmdb/log_{study_id}.tsv",
                sep="\t",
            )
            log_df = log_df[log_df["stored"]]
            log_df["sample_id"] = log_df["view"].apply(lambda x: f"{study_id}_{x}")
        df_list.append(log_df)
    log_df = pd.concat(df_list)
    working_ids = sorted(log_df["sample_id"].values.tolist())
    body_inference_ids = tf.data.Dataset.from_tensor_slices(working_ids).batch(
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
    inference_steps = len(working_ids) // batch_size + int(
        (len(working_ids) % batch_size) > 0.5
    )

    predictions = model.predict(io_inference_ds, steps=inference_steps, verbose=1)
    for i, task in enumerate(view_annotation_tasks):
        model_predictions[f"{task}_prediction"] = predictions[i]

    df = pd.DataFrame()
    df["sample_id"] = working_ids[: len(model_predictions[f"{task}_prediction"])]
    for task in view_annotation_tasks:
        df[f"{task}_prediction"] = np.argmax(
            model_predictions[f"{task}_prediction"], axis=1
        )
        df[f"{task}_prediction_probability"] = np.max(
            model_predictions[f"{task}_prediction"], axis=1
        )
    df.to_csv(output_file, index=False, sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lmdb_folder", type=str)
    parser.add_argument("--is_mgh", action="store_true")
    parser.add_argument("--movinet_ckpt_dir", type=str)
    parser.add_argument("--n_input_frames", type=int, default=1)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--pretrained_ckpt_dir", type=str)
    parser.add_argument("--skip_modulo", type=int, default=1)
    parser.add_argument("--study_df", type=str)
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
        pd.read_csv(args.study_df, sep='\t', dtype={'MRN': 'str'}),
        args.view_annotation_tasks,
    )