from tensorflow.keras.models import load_model

from ml4h.TensorMap import TensorMap, Interpretation
from ml4h.models.model_factory import get_custom_objects

custom_dict = get_custom_objects([])
ecg_model_file = '/home/sam/hypertuned_48m_16e_ecg_median_raw_10_autoencoder_256d/encoder_ecg_rest_median_raw_10.h5'
ecg_median_autoencoder_256d = TensorMap(
    'ecg_median_autoencoder_256d', Interpretation.EMBEDDING, shape=(256,),
    model=load_model(ecg_model_file, custom_objects=custom_dict),
)
