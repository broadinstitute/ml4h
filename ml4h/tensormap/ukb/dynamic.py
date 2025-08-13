from ml4h.TensorMap import TensorMap
from ml4h.tensormap.ukb.demographics import build_age_from_instances_all_ages_202401, \
    build_bmi_21001_from_instances_all_bmis_202401, build_bmi_23104_from_instances_all_bmis_202401

def make_ukb_dynamic_tensor_maps(desired_map_name: str, tensors: str) -> TensorMap:
    tensor_map_maker_fxns = [
        make_csv_maps,
    ]
    for map_maker_function in tensor_map_maker_fxns:
        desired_map = map_maker_function(desired_map_name, tensors)
        if desired_map is not None:
            return desired_map

def make_csv_maps(desired_map_name: str, tensors: str) -> TensorMap:
    for instance in [0,1,2,3]:
        # age
        name = f'age_{instance}_from_instances_all_ages_202401'
        if name == desired_map_name:
            return build_age_from_instances_all_ages_202401(tensors, instance)

        # bmi
        name = f'bmi_21001_{instance}_from_instances_all_bmis_202401'
        if name == desired_map_name:
            return build_bmi_21001_from_instances_all_bmis_202401(tensors, instance)

        name = f'bmi_23104_{instance}_from_instances_all_bmis_202401'
        if name == desired_map_name:
            return build_bmi_23104_from_instances_all_bmis_202401(tensors, instance)
