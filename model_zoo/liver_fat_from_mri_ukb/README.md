# Machine learning enables new insights into clinical significance of and genetic contributions to liver fat accumulation

Here we host two models for estimating liver fat from abdominal MRI. 
The liver fat percentage training data is from the returned liver fat values in the [UK Biobank field ID 22402](https://biobank.ctsu.ox.ac.uk/crystal/field.cgi?id=22402).  These values were only calculated for the echo protocol, so to infer liver fat from the ideal protocl we used a teacher/student modeling approach.

## Teacher Model
The teacher model was trained with abdominal MRIs acquired using the echo protocol.  
This model takes input of shape 160 x 160 x 10 and emits a scalar representing estimated liver fat percentage.
The input TensorMap is defined at `tensormap.ukb.mri.gre_mullti_echo_10_te_liver`.
The output TensorMap associated with these values is defined at `tensormap.ukb.mri.liver_fat`.
The keras model file is at [liver_fat_from_echo.h5](liver_fat_from_echo.h5) and the model architecture is shown below:
![](liver_fat_from_echo_teacher_model.png)


## Student Model
The teacher model made inferences on all available MRIs acquired with the echo protocol.  
This includes some individuals who also had abdominal MRI with the ideal protocol.
The student model was trained on these individuals, using the teacher model's inferences as truth data and abdominal MRIs acquired with the ideal protocol as input.  
This model takes input of shape 232 x 256 x 36 and also emits a scalar representing estimated liver fat percentage.
The input TensorMap is defined at `tensormap.ukb.mri.lms_ideal_optimised_low_flip_6dyn`.
The output TensorMap associated with these values is defined at `tensormap.ukb.mri.liver_fat_echo_predicted`. 
The keras model file is at [liver_fat_from_ideal.h5](liver_fat_from_ideal.h5) and the model architecture is shown below:
![](liver_fat_from_ideal_student_model.png)
