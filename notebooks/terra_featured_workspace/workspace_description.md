# Use ml4h to review and annotate clinical data and machine learning results

In this Terra workspace we demonstrate the notebooks used by clinicians and researchers to review and annotate both clinical data model inputs, such as phenotypes, ECGs and MRIs, and model outputs such as predicted left ventricular mass.

`ml4h` is a project aimed at using machine learning to model multi-modal cardiovascular
time series and imaging data. `ml4h` began as a set of tools to make it easy to work
with the UK Biobank on Google Cloud Platform and has since expanded to include other data sources
and functionality.

Please see https://github.com/broadinstitute/ml4h/ for more details on the full project.

----------------------------
## How long will it take to run? How much will it cost?
**Time:** It takes 1 minute to run each notebook. Spend as much or as little time as you like using the interactive visualizations to explore the data.

**Cost:** Using the default notebook configuration, the Terra notebook runtime charges are $0.20/hour for Google Cloud service costs. It should cost less than a quarter to run the notebooks.

----------------------------
## Get Started

1. Clone this workspace.
1. Run notebook `ml4h_setup.ipynb` to install the ml4h Python package and data visualization Jupyter extensions on your cloud environment.
1. Run the notebooks in "Playground Mode" to explore model inputs and outputs!

----------------------------
## Notebooks

* **review_model_results_interactive**: Use this notebook to perform interactive quality control (QC) of a simulated ECG and MRI prediction model.
* **review_one_sample_interactive**: Use this notebook to perform interactive quality control (QC) of per-patient multi-modal data for clinical machine learning models.
* **image_annotations_demo**: Use this notebook to annotate MRI images to create new input data for machine learning.
* **mnist_survival_analysis_demo**: In survival analysis, the aim is to predict when an event might occur, such as a heart attack, stroke, or the onset of heart feailure. This notebook uses the MNIST dataset to develop a toy model of survival analysis using ML4H.

### Cloud Environment

When you create your cloud environment, you can specify that it be [GPU-enabled](https://support.terra.bio/hc/en-us/articles/4403006001947).  While all of the notebooks above will run fine on CPUs, the `mnist_survival_analysis_demo.ipynb` notebook in particular, which trains several ML models, will benefit from using a GPU.

| Option | Value |
| --- | --- |
| Environment | Default 'application environment' (GATK, Python, R) |
| CPU Minimum | 4|
| Disk size Minimum | 50 GB |
| Memory Minimum | 15 GB |
| GPU | optional, but beneficial for model training|

----------------------------
## Next steps

* Try these notebooks on your own data.
* Read more about the ml4h project on  https://github.com/broadinstitute/ml4h/
* Ask questions https://github.com/broadinstitute/ml4h/issues
* Apply those resources to your own research!

---

### Contact information

You can also reach us on GitHub by [filing an issue](https://github.com/broadinstitute/ml4h/issues).

### License
Please see the BSD-3-Clause [license on GitHub](https://github.com/broadinstitute/ml4h/blob/master/LICENSE.TXT)

### Workspace Change Log
Please see the [pull request history](https://github.com/broadinstitute/ml4h/pulls?q=is%3Apr+) on GitHub.