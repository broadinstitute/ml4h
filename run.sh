#./scripts/tf.sh /home/${USER}/repos/ml/ml4cvd/recipes.py --mode train --tensors /home/${USER}/ukbb/ --input_tensors ecg_rest --output_tensors qt-interval --training_steps 3 --validation_steps 1 --test_steps 1 --batch_size 1 --epochs 16 --patience 12 --output_folder /home/${USER}/output/ --id ecg_rest_${USER}_rhythm --learning_rate 0.00002

#./scripts/tf.sh /home/${USER}/repos/ml/ml4cvd/recipes.py --mode train --tensors /home/erik/partners_ecg_hd5_paolo/ --input_tensors ecg_partners --output_tensors ecg_partners_read --inspect_model --training_steps 96 --validation_steps 32 --test_steps 16 --batch_size 32 --epochs 500 --patience 12 --test_modulo 0 --output_folder /home/${USER}/output/ --id ecg_partners__ecg_partners_read --learning_rate 0.00002

#./scripts/tf.sh /home/${USER}/repos/ml/ml4cvd/recipes.py --mode train --tensors /home/erik/partners_ecg_hd5_paolo/ --input_tensors ecg_partners --output_tensors ecg_partners_intervals --inspect_model --training_steps 96 --validation_steps 32 --test_steps 16 --batch_size 32 --epochs 20 --patience 12 --test_modulo 0 --output_folder /home/${USER}/output/ --id ecg_partners__ecg_partners_intervals --learning_rate 0.00002

./scripts/tf.sh /home/${USER}/repos/ml/ml4cvd/recipes.py --mode train --tensors /home/${USER}/partners_ecg_betatest/ --input_tensors ecg_partners --output_tensors supranodal_rhythms --inspect_model --training_steps 384 --validation_steps 32 --test_steps 16 --batch_size 32 --epochs 40 --patience 12 --test_modulo 0 --output_folder "/home/${USER}/Dropbox\ \(Partners\ HealthCare\)/partners_ecg/ml4cvd_results/" --id supranodal_rhythms --learning_rate 0.00002

