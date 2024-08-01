<!--Age had to be set as a categorical group in order for inference to work from my testing. I have set it so that a group labeled "patientage" or "age" will always be set to categorical for now. -->
<!--After cleaning up my code, I ran inference again on the resulting hdf5 files. The results were very slightly different (5 decimal places in). Is that normal, or a sign my efforts have changed the resulting hdf5 files? -->
<h1>Convert CSV Files to HDF5</h1>
This script is for converting CSV files into HDF5 format. It was originally created with 12-lead ECGs in mind. An additional file referred to as the diagnostic file can be included. The datasets and diagnostic files in which this method was originally used can be found at

<https://figshare.com/collections/ChapmanECG/4560497/2>
<!--I haven't ever gotten inference to work without the diagnostic file. So it may be necessary, not optional. -->
<h2>The CVS Files</h2>
CSV files to convert should be homogeneous in shape (all rows are the same length). <u>The 6th through 14th characters in each file's name should be the date the data was gathered on.</u>
<!--Inference seemed to expect a date to be present within each dataset. In the data I created this script for, the only date listed is within the filename. This is probably not a good solution, and I would like to implement a better one soon. But I'm not certain where the date would be located otherwise. -->
If the first row of the ECG files are labels for the columns, then the script should be run with hasHeaders set to True. Otherwise, it should be set to False. 
<h2>The Diagnostic File</h2>
The diagnostic file should be a CSV or other similar file where each row contains additional information (such as age, gender, atrial rate, presence of an illness or condition ect.) regarding each file in the CSV directory. This would likely be the location where truth values for machine learning studies would be stored. 
<h2>Parameters</h2>

**sourceDirectoryName**: The path to the directory containing the CSV files to be converted to HDF5 format. 

**diagnosticFile**: The path to an optional file containing additional information about every file in the source directory. The first row should act as labels for the data and the first column should be the patient's name/ID.

**datatype**: The format the CSV data in will be stored in the HDF5 files. Defaults to float.

**outputDirectoryName**: The directory the converted files will be stored in. Will generate a name using the source directory if none is provided.

**hasHeaders**: If the top row the CSV files are headers/labels for the data. 

**groupLabels**: A list containing the labels for the columns of the CSV files that will form the groups within the HDF5 files. If groupLabels is left empty, group labels will be generated from the file headers, or they will be labeled numerically if hasHeaders = False.
Will cause an error if the length of the list does not match the number of rows in the CSV files. 

**groupNamePrefix**: A string that is applied to the beginning of every non-categorical group. Useful if they are all to go in a certain subgroup or dataset. Defaults to "ukb_ecg_rest/"

**ageLabel**: The label corresponding to patient's age in the diagnostic file. Used to ensure that age is considered a categorical attribute. Defaults to 'PatientAge'.

Example Command: 
```
python hdf5_convert.py --sourceDirectoryName Path/To/CSVFiles --diagnosticFile Path/To/Diagnostic/File --datatype float --outputDirectoryName Path/To/Output/Directory --hasHeaders True --groupNamePrefix "/ukb_ecg_rest/" --groupLabels strip_I strip_II strip_III strip_aVR strip_aVL strip_aVF strip_V1 strip_V2 strip_V3 strip_V4 strip_V5 strip_V6
```

<h2>Running Inference After Conversion</h2>
Running inference using recipes.py in the ml4h repository expects the group ukb_ecg_rest/ecg_rest_text/ to be present in the HDF5 files. However this script is not programmed to add this group. In order to run inference on the resulting HD5 files these lines must be removed from the method "ecg_rest_from_file" in ecg.py:

```
ecg_interpretation = str(
    tm.hd5_first_dataset_in_group(
    hd5, 'ukb_ecg_rest/ecg_rest_text/',
    )[()],
)
if skip_poor and 'Poor data quality' in ecg_interpretation:
    raise ValueError(f'Poor data quality skipped by {tm.name}.')
```
<!--I would like to get this group added to the code instead. But I'm not certain what it is and for the very moment I'd like to get this cleaned up and documented without adding any new features.--> 

Once the files are ready, you will need to download a trained model. This project this program was created for used the ecg2af_12lead model. To download this model, run
```
gsutil cp gs://ml4h-terra/models/ECG2AF/ecg2af_12lead.h5
```

Then from the ml4h directory, run 
```
 ./scripts/tf.sh ${HOME}/ml4h/ml4h/recipes.py --mode infer --tensors (path/to/yourHD5Files) --input_tensors ecg.ecg_rest_mgb --output_tensors survival.mgb_afib_wrt_instance2 demographics.age_2_wide demographics.af_dummy demographics.sex_dummy  --tensormap_prefix ml4h.tensormap.ukb --id ecg_inference --output_folder inference/(output/folder) --model_file (path/to/downloaded/model/file)
```
with the parameters in parenthesis being replaced with the values relevant to you. A different command may be needed if you are lacking a diagnostics file.

After running inference, the resulting file may not include the truth values. In order to judge accuracy of your results, you will need to add the truth values back into the resulting files. The script add_truth.py will do this, but only for sex, age, and the presence of atrial fibrillation. It also requires the diagnostic files to have the relevant information stored in the expected way. See the relevant file for more information.
<!--Again, another quick fix I would like to be able to fix if possible.-->
