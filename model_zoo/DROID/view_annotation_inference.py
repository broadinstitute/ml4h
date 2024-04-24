import argparse
import logging

import tensorflow as tf

from echo_defines import category_dictionaries, view_annotation_tasks
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
    movinet_ckpt_dir,
    n_input_frames,
    output_file,
    pretrained_ckpt_dir,
    skip_modulo,
):
    input_dd = LmdbEchoStudyVideoDataDescriptionBWH(
        lmdb_folder,
        "image",
        [],
        n_input_frames,
        skip_modulo,
    )

    # Load the IDs
    log_df = pd.read_csv('/data/ewok/bwh_lmdbs/00003665_EVS0399897.lmdb/log_00003665_EVS0399897.tsv', sep='\t')
    log_df = log_df[log_df['stored']]
    log_df['sample_id'] = log_df['view'].apply(lambda x: f'00003665_EVS0399897_{x}')
    working_ids = sorted(log_df['sample_id'].values.tolist())
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
        path=movinet_ckpt_dir,
        input_shape=(n_input_frames, 224, 224, 3))
    model = create_classifier(
        encoder, 
        trainable=False, 
        input_shape=(n_input_frames, 224, 224, 3), 
        categories={
            f'annotation_{task}': len(category_dictionaries[task]) for task in view_annotation_tasks
            }
    )

    model.load_weights(pretrained_ckpt_dir)

    model_predictions = {}
    inference_steps = len(working_ids) // batch_size + int((len(working_ids) % batch_size) > 0.5)

    predictions = model.predict(io_inference_ds, steps=inference_steps, verbose=1)
    for i, task in enumerate(view_annotation_tasks):
        model_predictions[f'{task}_prediction'] = predictions[i]

    df = pd.DataFrame()
    df['sample_id'] = working_ids[:len(model_predictions[f'{task}_prediction'])]    
    for task in view_annotation_tasks:
        df[f'{task}_prediction'] = np.argmax(model_predictions[f'{task}_prediction'], axis=1)
        df[f'{task}_prediction_probability'] = np.max(model_predictions[f'{task}_prediction'], axis=1)
    df.to_csv(output_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lmdb_folder", type=str)
    parser.add_argument("--movinet_ckpt_dir", type=str)
    parser.add_argument("--n_input_frames", type=int, default=1)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--pretrained_ckpt_dir", type=str)
    parser.add_argument("--skip_modulo", type=int, default=1)

    args = parser.parse_args()

    main(
        args.batch_size,
        args.lmdb_folder,
        args.movinet_ckpt_dir,
        args.n_input_frames,
        args.output_file,
        args.pretrained_ckpt_dir,
        args.skip_modulo,
    )
