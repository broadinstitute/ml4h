# Includes
# External
import logging
import datetime
from typing import Any
from dateutil import relativedelta
import numpy as np
from scipy.ndimage import zoom
from keras.utils import to_categorical
# Metrics
from ml4cvd.metrics import sentinel_logcosh_loss, survival_likelihood_loss, pearson
from ml4cvd.metrics import per_class_recall, per_class_recall_3d, per_class_recall_4d, per_class_recall_5d
from ml4cvd.metrics import per_class_precision, per_class_precision_3d, per_class_precision_4d, per_class_precision_5d
# Defines
from ml4cvd.defines import DataSetType, CODING_VALUES_MISSING, TENSOR_MAP_GROUP_MISSING_CONTINUOUS, TENSOR_MAP_GROUP_CONTINUOUS
from ml4cvd.defines import EPS, JOIN_CHAR, IMPUTATION_RANDOM, IMPUTATION_MEAN, CODING_VALUES_LESS_THAN_ONE, MRI_SEGMENTED_CHANNEL_MAP
from ml4cvd.defines import MRI_FRAMES, MRI_SEGMENTED, MRI_TO_SEGMENT, MRI_ZOOM_INPUT, MRI_ZOOM_MASK, MRI_ANNOTATION_NAME, MRI_ANNOTATION_CHANNEL_MAP
# Utility
from ml4cvd.utility import normalize_zero_mean_std1, _is_equal_field

########################################
# TensorMap
########################################
class TensorMap(object):
    """TensorMap is the core structure used...
    """    
    # Static members
    # Todo: This information is too specific and should be refactored out.
    not_missing = 'not-missing'

    # Constructor
    def __init__(self,
                 name: str,
                 shape: [int, ...] = None,
                 group: str = None,
                 loss = None,
                 model = None,
                 metrics = None,
                 parents = None,
                 sentinel = None,
                 activation: str = None,
                 loss_weight: float = 1.0,
                 channel_map = None,
                 hd5_override = None,
                 dependent_map = None,
                 required_inputs = None,
                 normalization: [str, ...] = None,
                 annotation_units: int = 32,
                 imputation = None,
                 tensor_from_file = None,
                 dtype = None,
                 validator = None,
                 cacheable: bool = True):
        """TensorMap constructor.
        
        Arguments:
            name {str} -- Name of the tensor mapping
        
        Keyword Arguments:
            shape {[int, ...]} -- Tuple of integers specifying tensor shape
            group {[str]} -- String group of the tensor mapping
            loss {[type]} -- Loss function or str specifying pre-defined loss function
            model {[type]} -- Model for hidden layer tensor maps
            metrics {[type]} -- List of metric functions of strings
            parents {[type]} -- List of tensorMaps which must be attached to the graph before this one
            sentinel {[type]} -- If set, this value should never naturally occur in this TensorMap, it will be used for masking loss function
            activation {str} -- String specifying activation function
            loss_weight {float} -- Relative weight of the loss from this tensormap (default: 1.0)
            channel_map {[type]} -- Dictionary mapping strings indicating channel meaning to channel index integers
            hd5_override {[type]} -- Override default behavior of tensor_from_file
            dependent_map {TensorMap} -- TensorMap that depends on or is determined by this one
            required_inputs {[type]} -- List of TensorMaps that are required by this one, used by hidden layer TensorMaps
            normalization {[str, ...]} -- Dictionary specifying normalization values
            annotation_units {int} -- Size of embedding dimension for unstructured input TensorMap (default: 32)
            imputation {[type]} -- Method of imputation for missing values. Options are mean or random.
            tensor_from_file {[type]} -- Function that returns numpy array from hd5 file for this TensorMap
            dtype {[type]} -- DataSetType of TensorMap
            validator {[type]} -- [description] (default: {None})
            cacheable {bool} -- [description] (default: {True})
        """
        self.name = name
        self.loss = loss
        self.model = model
        self.shape = shape
        self.group = group
        self.metrics = metrics
        self.parents = parents
        self.sentinel = sentinel
        self.activation = activation
        self.loss_weight = loss_weight
        self.channel_map = channel_map
        self.hd5_override = hd5_override
        self.normalization = normalization
        self.dependent_map = dependent_map
        self.required_inputs = required_inputs
        self.annotation_units = annotation_units
        self.imputation = imputation
        self.tensor_from_file = tensor_from_file
        self.dtype = dtype
        self.validator = validator
        self.cacheable = cacheable

        # If the input shape is unspecified then we set the shape to be either
        # the tuple (k|channel map|,), where k is either 2 or 1 for continuous
        # multi-field data and all others, respectively.
        if self.shape is None:
            # Continuous multi-field.
            if self.is_multi_field_continuous_with_missing_channel():
                self.shape = (len(self.channel_map) * 2,)
            # All others input types.
            else:
                if self.channel_map is not None:
                    self.shape = (len(self.channel_map),)

        # Set the default activation functions when the input parameter for activation
        # is empty (None). The defaults are set to softmax and linear for categorical
        # and linear input data types, respectively.
        if self.activation is None:
            # Case 1: input type is categorical -> use a softmax acivation function.
            if self.is_categorical_any():
                self.activation = 'softmax'
            # Case 2: input type is continuous -> use a linear activation function.
            elif self.is_continuous_any():
                self.activation = 'linear'

        # Set the default loss functions when the input parameter for loss is empty (None).
        # By default, the loss functon will be set as follows given the input data type:
        #    1) categorical data: categorical cross-entropy
        #    2) continuous data AND is not a sentinel: 
        #    3) continuous data: mean square error (mse)
        #    4) proportional hazard ratio: survival likelihood loss
        #    5) fallback: mean square error (mse)
        if self.loss is None:
            # Any categorical input data
            if self.is_categorical_any():
                self.loss = 'categorical_crossentropy'
            # Continuous input data while the data is NOT a sentinel node.
            elif self.is_continuous() and self.sentinel is not None:
                self.loss = sentinel_logcosh_loss(self.sentinel)
            elif self.is_continuous():
                self.loss = 'mse'
            elif self.is_proportional_hazard():
                # Use the Keras subroutine for survival likelihood loss.
                self.loss = survival_likelihood_loss(self.shape[0]//2)
                self.activation = 'sigmoid'
            else: # Fallback to mean-square error.
                self.loss = 'mse'

        # Set the default metric when the input parameter is empty (None). If the input data
        # is categorical then set the metrics to a dimension-specific [precision, recall]-tuple.
        # If the input data is continuous we set the metrics to be Pearson's correlation coefficient.
        # If neither of the above conditions are true then no fallback metrics will be used.
        if self.metrics is None:
            if self.is_categorical_any():
                self.metrics = ['categorical_accuracy']
                # Depending on the length of the input shape (2D, 3D, 4D, or 5D), we additionally
                # add precision and recall to the metrics list.
                if len(self.shape) == 1: # 2D
                    self.metrics += per_class_precision(self.channel_map)
                    self.metrics += per_class_recall(self.channel_map)
                elif len(self.shape) == 2: # 3D
                    self.metrics += per_class_precision_3d(self.channel_map)
                    self.metrics += per_class_recall_3d(self.channel_map)
                elif len(self.shape) == 3: # 4D
                    self.metrics += per_class_precision_4d(self.channel_map)
                    self.metrics += per_class_recall_4d(self.channel_map)
                elif len(self.shape) == 4: # 5D
                    self.metrics += per_class_precision_5d(self.channel_map)
                    self.metrics += per_class_recall_5d(self.channel_map)
            elif self.is_continuous_any():
                self.metrics = [pearson]
            else:
                self.metrics = []

        # If no validation function is provided (None) then we define this function as
        # an anonymous labmda function f(x) -> x. This function will evaluate parity between
        # the input and the input (input == input) and will thus ascertain correctness in 
        # the default case.
        if self.validator is None:
            self.validator = lambda tm, x: x

        # If the input type is not a categorical type and no normalization procedure 
        # have been selected then we set the normalization function to "zero_mean_std1".
        if self.normalization is None and not self.is_categorical_any():
            self.normalization = {'normalize_zero_mean_std1': 1.0}

        ##############################
        # DISCONTINUED
        # If method is passed to load HDF5 tensors from file we will default to
        # the "_hdf5_load_tensor" subroutine. Currently these functions
        # _MUST_ match the signature (TensorMap, hdf5, {dependents})!
        # if self.tensor_from_file is None:
        #     self.tensor_from_file = _hdf5_load_tensor
            
        # END CTOR

    # Hash this class such that H({name, shape, group}) -> Z.
    def __hash__(self):    
        return hash((self.name, self.shape, self.group))

    def __eq__(self, other) -> bool:
        # Ascertain that the other instance is also of type TensorMap. Return a NotImplemented
        # exception if that is the case.
        if not isinstance(other, TensorMap):
            return NotImplemented
        else:
            # Iterate over the attribute in this instance and compare it agains the attributes
            # in the other instance. If a mismatch is found then return False otherwise return True.
            for (self_field, self_value), (other_field, other_value) in zip(self.__dict__.items(), other.__dict__.items()):
                # Mismatch between attributes.
                if self_field != other_field:
                    return False
                # TODO
                # Critical mismatch between the two instances.
                if not _is_equal_field(self_value, other_value):
                    logging.debug(f"Comparing two '{self.name}' tensor maps: "
                                  f"'{self_field}' values '{self_value}' and '{other_value}' are not equal.")
                    return False

            # The two class instances are equal.
            return True

    ########################################
    # Functions
    ########################################
    # Imputes missing values using either randomness or using means. This subroutines 
    # requires that the input data is normalized.
    def impute(self):
        # No normalization has been used.
        if self.normalization is None:
            return ValueError('Imputation requires normalization.')
        # Use random value permutation.
        if self.is_imputation_random():
            return np.random.normal(1)
        # Impute using mean values.
        elif self.is_imputation_mean():
            return 0
        # Unknown imputation method. Return a ValueError.
        else:
            return ValueError('Imputation method unknown.')

    # Rescale the input data.
    def rescale(self, np_tensor):
        """Rescales the target input tensor given pre-specified normalization procedure
        If no normalization function has been provided then simply return the
        input value as-is.

        Arguments:
            np_tensor {[type]} -- Input tensor
        
        Returns:
            np_tensor -- Either a normalized tensor or the input tensor
        """        
        if self.normalization is None:
            return np_tensor
        # If both the keywords mean and std can be found in the normalization list
        # then first normalize standard deviation and then by mean.
        elif 'mean' in self.normalization and 'std' in self.normalization:
            np_tensor = np.array(np_tensor) * self.normalization['std']
            np_tensor = np.array(np_tensor) + self.normalization['mean']
            return np_tensor
        # If the procedure is zero mean and standard deviation one.
        elif 'zero_mean_std1' in self.normalization:
            return normalize_zero_mean_std1(np_tensor)
        # All other unknown cases we simply return the input tensor as-is.
        else:
            return np_tensor

    def normalize_and_validate(self, np_tensor):
        """Normalized and validate the target input tensor. The target validation
        and normalization functions must be predefined.
        
        Arguments:
            np_tensor {[type]} -- Input Numpy tensor.
        
        Returns:
            [type] -- Returns the normalized and validated iput tensor.
        """        
        # Execute the validator function on the given input data. This function
        # defaults to a lambda f(x) -> x if none has been provided.
        # Todo: the validator is not a predicate evaluator and nothing is used with the return data!
        self.validator(self, np_tensor)

        # Perform any desired normalization on the input data.
        # Note that normalization
        if self.normalization is None:
            return np_tensor
        
        # If the normalization method list include zero-mean and 1-standard deviation.
        if 'zero_mean_std1' in self.normalization:
            return normalize_zero_mean_std1(np_tensor)
        
        # If both mean and std are found in the normalization list.
        # Note that it is possible to pass both these arguments and the
        # "zero_mean_std_1" string i.e. ["zero_mean_std_1", "std", "mean"].
        if 'mean' in self.normalization and 'std' in self.normalization:
            not_missing_in_channel_map = False

            # Channel map argument have been set.
            if self.channel_map is not None:
                not_missing_in_channel_map = self.not_missing in self.channel_map
            
            # Only proceed with (mean, std)-normalization if the input data is
            # is continuous and is not missing from the channel map.
            if self.is_continuous() and not_missing_in_channel_map:
                # Iterate over the input tensor and check if the target
                # channel_map is missing or not. If it is, then continue.
                for i in range(0, len(np_tensor)):
                    if self.channel_map[self.not_missing] == i:
                        continue
                    # If the not-missing channel exists in the channel_map and it is marked as "missing" (value of 0)
                    # and the data itself is 0, then overwrite the value with a draw from a N(0,1)
                    if np_tensor[self.channel_map[self.not_missing]] == 0 and np_tensor[i] == 0:
                        np_tensor[i] = np.random.normal(1)
                    elif np_tensor[i] == 0:
                        np_tensor[i] -= self.normalization['mean']
                        np_tensor[i] /= (self.normalization['std'] + EPS) # Add small value to prevent 0-division
            # In all other cases, for example if the input data is categorical.
            else:
                np_tensor -= self.normalization['mean']
                np_tensor /= (self.normalization['std'] + EPS) # Add small value to prevent 0-division
            return np_tensor

    ########################################
    # Setters and getters
    ########################################
    def output_name(self, prefix: str = "output", delimiter: str = JOIN_CHAR):
        """Computes an output name given the prefix string and a delimiter such that the output
        string equals PREFIX DELIMITER NAME or PREFIX DELIMITER NAME DELIMITER GROUP if the grouping
        parameter is used. The prefix defaults to "output" and the delimiter defaults to ML4CVD_DELIMITER.
        
        Keyword Arguments:
            prefix {str} -- Output prefix string (default: "output")
            delimiter {str} -- Delimiter string (default: JOIN_CHAR)
        
        Returns:
            str -- A formatted output string
        """        
        if self.group is None:
            return delimiter.join([prefix, self.name])
        else:
            return delimiter.join([prefix, self.name, self.group])

    # Getter for a parse input string given a delimter and a prefix. The prefix
    # defaults to "input" and the delimiter defaults to ML4CVD_DELIMITER.
    def input_name(self, prefix: str = "input", delimiter: str = JOIN_CHAR):
        """Computes an input name given the prefix string and a delimiter such that the input
        string equals PREFIX DELIMITER NAME or PREFIX DELIMITER NAME DELIMITER GROUP if the grouping
        parameter is used. The prefix defaults to "input" and the delimiter defaults to ML4CVD_DELIMITER.
        
        Keyword Arguments:
            prefix {str} -- Input prefix string (default: "input")
            delimiter {str} -- Delimiter string (default: JOIN_CHAR)
        
        Returns:
            str -- A formatted input string
        """        
        if self.group is None:
            return delimiter.join([prefix, self.name])
        else:
            return delimiter.join([prefix, self.name, self.group])

    ########################################
    # Predicate evaluations
    ########################################
    # Imputation predicates.
    def is_imputation_random(self) -> bool:
        return self.is_multi_field_continuous() and self.imputation == IMPUTATION_RANDOM

    def is_imputation_mean(self) -> bool:
        return self.is_multi_field_continuous() and self.imputation == IMPUTATION_MEAN

    # Data type predicates.
    def is_categorical_any(self) -> bool:
        """Evaluate if the group type is one the special cases: 1) categorical index, 
        2) categorical date, 3) ECG categorical interpretation, OR have its dtype set to
        categorical (DataSetType.CATEGORICAL). Returns the predicate evaluation as a Boolean.
        
        Returns:
            bool -- Returns True if any categorical value is used or False otherwise.
        """        
        return self.is_categorical_index() or self.is_categorical() or self.is_categorical_date() or self.is_categorical_flag() or self.is_ecg_categorical_interpretation() or self.dtype == DataSetType.CATEGORICAL

    def is_continuous_any(self) -> bool:
        return self.is_continuous() or self.is_diagnosis_time()

    def is_categorical_any_with_shape_len(self, length) -> bool:
        return self.is_categorical_any() and len(self.shape) == length

    def is_categorical(self) -> bool:
        return self.group == 'categorical'

    def is_categorical_index(self) -> bool:
        return self.group == 'categorical_index'

    def is_categorical_date(self) -> bool:
        return self.group == 'categorical_date'

    def is_categorical_flag(self) -> bool:
        return self.group == 'categorical_flag'

    def is_continuous(self) -> bool:
        return self.group == 'continuous' or self.dtype == DataSetType.CONTINUOUS

    def is_multi_field_continuous(self) -> bool:
        return self.is_multi_field_continuous_with_missing_channel() or self.group == TENSOR_MAP_GROUP_CONTINUOUS

    def is_root_array(self) -> bool:
        return self.group == 'root_array'

    def is_multi_field_continuous_with_missing_channel(self) -> bool:
        return self.group == TENSOR_MAP_GROUP_MISSING_CONTINUOUS

    def is_hidden_layer(self) -> bool:
        return self.group == 'hidden_layer' and self.model is not None

    def is_proportional_hazard(self) -> bool:
        return self.group == 'proportional_hazard'

    def is_diagnosis_time(self) -> bool:
        return self.group == 'diagnosis_time'

    def is_ecg_rest(self) -> bool:
        return self.group == 'ecg_rest'

    def is_ecg_categorical_interpretation(self) -> bool:
        return self.group == 'ecg_categorical_interpretation'

    def is_ecg_bike(self) -> bool:
        return self.group == 'ecg_bike'

    def is_ecg_bike_recovery(self) -> bool:
        return self.group == 'ecg_bike_recovery'

    def is_ecg_text(self) -> bool:
        return self.group == 'ecg_text'