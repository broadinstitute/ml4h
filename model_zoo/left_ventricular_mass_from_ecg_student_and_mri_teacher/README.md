# Deep Learning to Predict Cardiac Magnetic Resonance-Derived Left Ventricular Mass and Hypertrophy from 12-Lead Electrocardiograms

This folder contains models and code supporting the work described in [this paper](https://www.ahajournals.org/doi/10.1161/CIRCIMAGING.120.012281?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%20%200pubmed) published in the journal "Circulation: Cardiovascular Imaging". 

# LVM-AI
Left Ventricular Mass-Artificial Intelligence (LVM-AI) is a one-dimensional convolutional neural network trained to predict CMR-derived LV mass using 12-lead ECGs. LVM-AI was trained within 32239 individuals from the UK Biobank with paired CMR and 12-lead ECG. It was provided with the entire 10 seconds of the 12-lead ECG waveform as well as participant age, sex, and BMI. 
LVM-AI was evaluated in a UK Biobank test set as well as an external health care–based Mass General Brigham (MGB) dataset. In both test sets, LVM-AI was compared to with traditional ECG-based rules for diagnosing CMR-derived left ventricular hypertrophy. Associations between LVM-AI predicted LV mass index and incident cardiovascular events were tested in the UK Biobank and a separate MGB-based ambulatory cohort (MGB outcomes)
![Overview of the training and test samples](TrainingAndTestSets.jpg)
When compared with any ECG rule, LVM-AI demonstrated similar LVH discrimination in the UK Biobank (LVM-AI c-statistic 0.653 [95% CI, 0.608 -0.698] versus any ECG rule c-statistic 0.618 [95% CI, 0.574 -0.663], P=0.11) and superior discrimination in MGB (0.621; 95% CI, 0.592 -0.649 versus 0.588; 95% CI, 0.564 -0.611, P=0.02). 


# Models 
Three pre-trained models are included here:
The model `ecg_rest_raw_age_sex_bmi_lvm_asymmetric_loss.h5` takes as input a 12 Lead resting ECG, as well as age, sex and BMI and has two outputs: one which regresses the left ventricular mass, and a second which gives a probability of left ventricular hypertrophy. This model was trained with the asymmetric loss described in the paper.  
The model `ecg_rest_raw_lvm_asymmetric_loss.h5` takes only an ECG as input and regresses left ventricular mass. This model was also trained with the asymmetric loss.
The third model, `ecg_rest_raw_lvm_symmetric_loss.h5` takes only an ECG as input and regresses left ventricular mass. This model was trained with the symmetric logcosh loss.  The raw voltage values from the ECG are normalized by dividing by 2000 prior to being input to the model.