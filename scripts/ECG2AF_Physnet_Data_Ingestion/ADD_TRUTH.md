<h1>AddTruthValues</h1>
This is a script to add missing truth values to the results of running inference using recipes.py.


<h2>The Diagnostics File</h2>

This script requires a diagnostics file similar to the one located [here](https://figshare.com/collections/ChapmanECG/4560497/2).
The first column of each row must contain the filename of the file that correlates to the patient represented by that row. The fields containing the presence of atrial fibrillation, the patients age, and sex must be in the 2nd, 4th and 5th columns respectively. The presence of atrial fibrillation is detected by the string "AFIB" being present in the 2nd column.
<!-- Again, currently needs a very specific setup. But it should work with the provided diagnostics.xlsx. I want to just get the first version cleaned up and documented before making it more universally applicable. Adding the ability to specify what the columns containing the truth values are named should be an option later. -->

<h2>Parameters</h2>

**inference_file**: The path to the tsv file received by running inference on your data using recipes.py.

**diagnostic_file**: The path to the diagnostic file containing the truth values for your data.

**output_file_name**: The name you would like the file with the truth values added in to have. Defaults to the input filename with "_added_truth_values" appended to it.

**remove_nan**: If the script should remove any entries in which the predicted confidence in the presence of atrial fibrillation is 'nan'. Defaults to true.


<h2>Example Command</h2>

```
python add_truth.py --inference_file path/to/inference.tsv --diagnostic_file path/to/Diagnostics.xlsx
```

