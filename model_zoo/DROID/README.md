# DROID (Dimensional Reconstruction of Imaging Data)

DROID is a 3-D convolutional neural network modeling approach for echocardiographic view
classification and quantification of LA dimension, LV wall thickness, chamber diameter and
ejection fraction.

The DROID echo movie encoder is based on the 
[MoViNet-A2-Base](https://tfhub.dev/tensorflow/movinet/a2/base/kinetics-600/classification/3) 
video classification model. MoViNet was fine-tuned in a supervised fashion to produce two
specialized encoders:
- DROID-LA
  - input views: PLAX, A4C, A2C
  - output predictions: LA A/P
- DROID-LV
  - input views: PLAX, A4C, A2C
  - output predictions: LVEF, LVEDD, LVESD, IVS, PWT

Multi-instance attention heads were then train to integrate up to 40 view encodings to predict
a single measurement of each type per echo study.