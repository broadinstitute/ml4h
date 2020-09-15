/home/pdiachil/ml/scripts/tf.sh /home/pdiachil/ml/ml4h/recipes.py \
        --mode explore \
        --input_tensors p_axis_raw p_duration_raw p_offset_raw p_onset_raw pp_interval_raw pq_interval_raw \
                        q_offset_raw q_onset_raw qrs_complexes_raw qrs_duration_raw qrs_num_raw qt_interval_raw qtc_interval_raw \
                        r_axis_raw rr_interval_raw ventricular_rate_raw t_offset_raw t_axis_raw \
        --tensors /mnt/disks/ecg-rest-37k-tensors/2019-11-04 \
        --tensormap_prefix ml4h.tensormap.ukb.ecg \
        --output_folder /home/pdiachil/formarcus \
        --id explore_ecg_features