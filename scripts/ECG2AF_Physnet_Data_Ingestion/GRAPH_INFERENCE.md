<h1>Graph_Inference</h1>
This script creates ROC and PR plots of a model's predictions of the patient's sex and the presence/absence of atrial fibrillation. In addition a scatterplot comparing the models predicted ages and actual ages. In order to run it you will need the TSV file from running inference using recipes.py, or a CSV file with similar data. The resulting graphs will go in a folder specified by the parameter "output_file_path".


<h2>Parameters</h2>

**inference_file**: The path to the tsv file received by running inference on your data using recipes.py.

**model**: The name of the model used for inference.

**output_file_path**: File path where the completed graphs should go. Defaults to './figures/'. 

**age_threshhold**: A lower bound of which any patients below this age will not show up on the age graph. (But will still be accounted for in the gender and AFIB graphs.)


<h2>Example Command</h2>

```
python graph_inference.py --inference_file /path/to/inference.csv --model ECG2AF --output_file_path "graphs/"
```



