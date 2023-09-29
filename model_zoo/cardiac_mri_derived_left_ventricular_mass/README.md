# Deep learning to estimate cardiac magnetic resonance–derived left ventricular mass
Within participants of the UK Biobank prospective cohort undergoing CMR, we trained 2 convolutional neural networks to estimate LV mass. The first (ML4H<sub>reg</sub>) performed regression informed by manually labeled LV mass (available in 5065 individuals), while the second (ML4Hseg) performed LV segmentation informed by InlineVF (version D13A) contours. 
# ML4H<sub>reg</sub>
The first model was a 3D convolutional neural network regressor ML4H<sub>reg</sub> trained with the manually annotated LV mass estimates provided by Petersen and colleagues to optimize the log cosh loss function, which behaves like L2 loss for small values and L1 loss for larger values: 
![Lreg](attachment:Lreg) 
Here batch size, N, is 4 random samples from the training set of 3178 after excluding testing and validation samples from the total 5065 CMR images with LV mass values included in P.
# ML4H<sub>seg</sub>
ML4H<sub>seg</sub>, is a 3D semantic
segmenter. To facilitate model development in the absence of hand-labeled segmentations, we trained with the InlineVF
contours to minimize Lseg; the per-pixel cross-entropy between the label and the model’s prediction. ![LSeg.png](attachment:LSeg.png)
Here the batch size, N, was 4 from the total set of 33,071. Height, H, and width, W, are 256 voxels and there was a
maximum of 13 Z slices along the short axis. There is a channel for each of the 3 labels, which were one-hot encoded in the training data, InlineVF (IVF), and probabilistic values from the softmax layer of ML4H<sub>seg</sub>. Segmentation architectures used U-Net-style long-range connections between early convolutional layers and deeper layers. Since not all CMR images used the same pixel dimensions, we built models to incorporate pixel size values with their fully connected layers before making predictions.