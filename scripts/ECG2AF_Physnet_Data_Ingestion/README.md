<h1>Usage instructions </h1>

This project was initially to run the ecg2af_12lead model on data from the study  “A large scale 12-lead electrocardiogram database for arrhythmia study”. The datasets and diagnostic files which were used can be found at <https://figshare.com/collections/ChapmanECG/4560497/2>.

Inference was run on both the standard and denoised datasets and supplemented by the "Diagnostics.xlsx" file. If you require further info and additional parameters, please see the markdown files of the individual scripts in this folder.




<h2>Converting CSV files to HD55</h2>
CVS files are converted to HDF5 files using "hdf5_convert.py" in order to run it, use the following command. Replace the placeholder values with the locations of the relevant files and directories. If you are converting the denoised dataset, then "--hasHeaders" should additionally be set to "False".

```
python hdf5_convert.py --sourceDirectoryName (Path/To/CSVFiles) --diagnosticFile (Path/To/Diagnostic/File) --datatype float --outputDirectoryName (Path/To/Output/Directory) --hasHeaders True --groupNamePrefix "/ukb_ecg_rest/" --groupLabels strip_I strip_II strip_III strip_aVR strip_aVL strip_aVF strip_V1 strip_V2 strip_V3 strip_V4 strip_V5 strip_V6
```


<h2>Running Inference After Conversion</h2>
Running inference using recipes.py in the ml4h repository expects the group ukb_ecg_rest/ecg_rest_text/ to be present in the HDF5 files. However this group is not present in the converted files. Therefore, in order to run inference on the resulting HD5 files these lines must be removed from the method "ecg_rest_from_file" in ecg.py:

```
ecg_interpretation = str(
   tm.hd5_first_dataset_in_group(
   hd5, 'ukb_ecg_rest/ecg_rest_text/',
   )[()],
)
if skip_poor and 'Poor data quality' in ecg_interpretation:
   raise ValueError(f'Poor data quality skipped by {tm.name}.')
```

<!--I would like to get this group added to the code instead if at all possible.-->


Once the files are ready, you will need to download a trained model. The project this program was created for used the ecg2af_12lead model. To download this model, run
```
gsutil cp gs://ml4h-terra/models/ECG2AF/ecg2af_12lead.h5
```


Then from within the ml4h directory, run
```
./scripts/tf.sh ${HOME}/ml4h/ml4h/recipes.py --mode infer --tensors (path/to/yourHD5Files) --input_tensors ecg.ecg_rest_mgb --output_tensors survival.mgb_afib_wrt_instance2 demographics.age_2_wide demographics.af_dummy demographics.sex_dummy  --tensormap_prefix ml4h.tensormap.ukb --id ecg_inference --output_folder inference/(output/folder) --model_file (path/to/downloaded/model/file)
```
with the parameters in parenthesis being replaced with the values relevant to you.


<h2>Adding truth values to inference files</h2>
After running inference, the resulting file may not include the truth values. In order to judge accuracy of your results, you will need to add the truth values back into the resulting files. The script add_truth.py will do this. To do this,  run:

```
python add_truth.py --inference_file (path/to/inference.tsv) --diagnostic_file (path/to/Diagnostics.xlsx)
```
This script currently requires a specifically setup diagnostic file to run properly. But will work on the Diagnostics.xlsx file downloaded from the link above.


<h2>Graphing inference Results</h2>
In order to create graphs of the resulting inference data, run the following command with relevant parameters replaced with your desired values.

```
python graph_inference.py --inference_file (/path/to/inference.csv) --model ECG2AF --output_file_path "graphs/"
```


The created plots will be in the folder specified in the "--output_file_path" parameter and will consist of ROC and PR plots of a model's predictions of the patient's sex and the presence/absence of atrial fibrillation. In addition a scatterplot comparing the models predicted ages and actual ages.