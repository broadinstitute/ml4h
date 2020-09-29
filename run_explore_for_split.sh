id=ecg_vae_cohort_09-29
pip install -e . 
python ml4cvd/recipes.py --mode explore --input_tensors partners_ecg_4096_no_norm partners_ecg_age_newest partners_ecg_rate_md_newest partners_ecg_qrs_md_newest partners_ecg_pr_md_newest partners_ecg_qt_md_newest --num_workers 10 --id ${id} --output_folder /storage/ndiamant/ml/explorations/ --tensors /storage/shared/ecg/mgh --plot_hist True
