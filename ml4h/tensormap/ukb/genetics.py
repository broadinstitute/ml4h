import numpy as np
from ml4h.TensorMap import TensorMap, Interpretation
from ml4h.defines import StorageType
from ml4h.metrics import weighted_crossentropy


diploid_cm = {'homozygous_reference': 0, 'heterozygous': 1, 'homozygous_variant': 2}
rs3829740 = TensorMap('rs3829740', Interpretation.CATEGORICAL, channel_map=diploid_cm)
rs2234962 = TensorMap('rs2234962', Interpretation.CATEGORICAL, channel_map=diploid_cm)
rs2042995 = TensorMap('rs2042995', Interpretation.CATEGORICAL, channel_map=diploid_cm)

rs3829740_weighted = TensorMap('rs3829740', Interpretation.CATEGORICAL, channel_map=diploid_cm, loss=weighted_crossentropy([1, 1, 1.5], 'rs3829740'))
rs2234962_weighted = TensorMap('rs2234962', Interpretation.CATEGORICAL, channel_map=diploid_cm, loss=weighted_crossentropy([.8, 1, 1.5], 'rs2234962'))
rs2042995_weighted = TensorMap('rs2042995', Interpretation.CATEGORICAL, channel_map=diploid_cm, loss=weighted_crossentropy([.6, 1.5, 2], 'rs2042995'))

# KCNJ5 = TensorMap('KCNJ5', Interpretation.CONTIG, channel_map={'position': 0, 'genotype': 1})
# KCNH2 = TensorMap('KCNH2', Interpretation.CONTIG, channel_map={'position': 0, 'genotype': 1})
# SCN5A = TensorMap('SCN5A', Interpretation.CONTIG, channel_map={'position': 0, 'genotype': 1})
# TTN = TensorMap('TTN', Interpretation.CONTIG, channel_map={'position': 0, 'genotype': 1})

akap9_lof = TensorMap('AKAP9', Interpretation.CATEGORICAL, channel_map={'no_akap9_lof': 0, 'akap9_lof': 1})
dsc2_lof = TensorMap('DSC2', Interpretation.CATEGORICAL, channel_map={'no_dsc2_lof': 0, 'dsc2_lof': 1})
ryr2_lof = TensorMap('RYR2', Interpretation.CATEGORICAL, channel_map={'no_ryr2_lof': 0, 'ryr2_lof': 1})
ttn_lof = TensorMap('ttn_lof', Interpretation.CATEGORICAL, channel_map={'no_ttn_lof': 0, 'ttn_lof': 1})

TTN_LOF = TensorMap('TTN', Interpretation.CATEGORICAL, path_prefix='categorical', storage_type=StorageType.CATEGORICAL_INDEX, channel_map={'no_TTN_LOF': 0, 'TTN_LOF': 1})
LMNA_LOF = TensorMap('LMNA', Interpretation.CATEGORICAL, path_prefix='categorical', storage_type=StorageType.CATEGORICAL_INDEX, channel_map={'no_LMNA_LOF': 0, 'LMNA_LOF': 1})
PKP2_LOF = TensorMap('PKP2', Interpretation.CATEGORICAL, path_prefix='categorical', storage_type=StorageType.CATEGORICAL_INDEX, channel_map={'no_PKP2_LOF': 0, 'PKP2_LOF': 1})
SCN5A_LOF = TensorMap('SCN5A', Interpretation.CATEGORICAL, path_prefix='categorical', storage_type=StorageType.CATEGORICAL_INDEX, channel_map={'no_SCN5A_LOF': 0, 'SCN5A_LOF': 1})
KCNQ1_LOF = TensorMap('KCNQ1', Interpretation.CATEGORICAL, path_prefix='categorical', storage_type=StorageType.CATEGORICAL_INDEX, channel_map={'no_KCNQ1_LOF': 0, 'KCNQ1_LOF': 1})
KCNH2_LOF = TensorMap('KCNH2', Interpretation.CATEGORICAL, path_prefix='categorical', storage_type=StorageType.CATEGORICAL_INDEX, channel_map={'no_KCNH2_LOF': 0, 'KCNH2_LOF': 1})
KCNJ5_LOF = TensorMap('KCNJ5', Interpretation.CATEGORICAL, path_prefix='categorical', storage_type=StorageType.CATEGORICAL_INDEX, channel_map={'no_KCNJ5_LOF': 0, 'KCNJ5_LOF': 1})
MYBPC3_LOF = TensorMap('MYBPC3', Interpretation.CATEGORICAL, path_prefix='categorical', storage_type=StorageType.CATEGORICAL_INDEX, channel_map={'no_MYBPC3_LOF': 0, 'MYBPC3_LOF': 1})

TTN_LOF_weighted = TensorMap(
    'TTN', Interpretation.CATEGORICAL, path_prefix='categorical', loss=weighted_crossentropy([0.2, 15.0]),
    storage_type=StorageType.CATEGORICAL_INDEX, channel_map={'no_TTN_LOF': 0, 'TTN_LOF': 1},
)

ttntv = TensorMap(
    'has_ttntv',  Interpretation.CATEGORICAL, path_prefix='categorical', storage_type=StorageType.CATEGORICAL_INDEX,
    channel_map={'no_ttntv': 0, 'has_ttntv': 1},
)

bsa_mosteller = TensorMap('bsa_mosteller',  Interpretation.CONTINUOUS, normalization={'mean': 1.8894831981880114, 'std': 0.22169301057810176}, loss='logcosh', channel_map={'bsa_mosteller': 0})
bsa_dubois = TensorMap('bsa_dubois',  Interpretation.CONTINUOUS, normalization={'mean': 1.8671809970639703, 'std': 0.20913930961120797}, loss='logcosh', channel_map={'bsa_dubois': 0})


genetic_pca_1 = TensorMap(
    '22009_Genetic-principal-components_0_1', Interpretation.CONTINUOUS, path_prefix='continuous', normalization={'mean': -0.014422761536727896, 'std': 10.57799283718005},
    loss='logcosh', channel_map={'22009_Genetic-principal-components_0_1': 0},
)
genetic_pca_2 = TensorMap(
    '22009_Genetic-principal-components_0_2', Interpretation.CONTINUOUS, path_prefix='continuous', #normalization={'mean': -0.014422761536727896, 'std': 10.57799283718005},
    loss='logcosh', activation='linear', channel_map={'22009_Genetic-principal-components_0_2': 0},
)
genetic_pca_3 = TensorMap(
    '22009_Genetic-principal-components_0_3', Interpretation.CONTINUOUS, path_prefix='continuous', #normalization={'mean': -0.014422761536727896, 'std': 10.57799283718005},
    loss='logcosh', activation='linear', channel_map={'22009_Genetic-principal-components_0_3': 0},
)
genetic_pca_4 = TensorMap(
    '22009_Genetic-principal-components_0_4', Interpretation.CONTINUOUS, path_prefix='continuous', #normalization={'mean': -0.014422761536727896, 'std': 10.57799283718005},
    loss='logcosh', activation='linear', channel_map={'22009_Genetic-principal-components_0_4': 0},
)
genetic_pca_5 = TensorMap(
    '22009_Genetic-principal-components_0_5', Interpretation.CONTINUOUS, path_prefix='continuous', #normalization={'mean': -0.014422761536727896, 'std': 10.57799283718005},
    loss='logcosh', activation='linear', channel_map={'22009_Genetic-principal-components_0_5': 0},
)
genetic_pca_all5 = TensorMap(
    'genetic_pca_all5', Interpretation.CONTINUOUS, path_prefix='continuous', normalization={'mean': -0.014422761536727896, 'std': 10.57799283718005},
    loss='logcosh', annotation_units=5, shape=(5,), activation='linear',
    channel_map={
        '22009_Genetic-principal-components_0_0': 0, '22009_Genetic-principal-components_0_1': 1,
        '22009_Genetic-principal-components_0_2': 2, '22009_Genetic-principal-components_0_3': 3,
        '22009_Genetic-principal-components_0_4': 4,
    },
)

genetic_caucasian = TensorMap(
    'Genetic-ethnic-grouping_Caucasian_0_0', Interpretation.CATEGORICAL, path_prefix='categorical', storage_type=StorageType.CATEGORICAL_FLAG,
    channel_map={'no_caucasian': 0, 'Genetic-ethnic-grouping_Caucasian_0_0': 1},
)

genetic_caucasian_weighted = TensorMap(
    'Genetic-ethnic-grouping_Caucasian_0_0', Interpretation.CATEGORICAL, path_prefix='categorical', storage_type=StorageType.CATEGORICAL_FLAG,
    channel_map={'no_caucasian': 0, 'Genetic-ethnic-grouping_Caucasian_0_0': 1}, loss=weighted_crossentropy([10.0, 1.0], 'caucasian_loss'),
)
