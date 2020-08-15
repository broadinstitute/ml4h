# Imports: standard library
from typing import Dict

# Imports: first party
from ml4cvd.TensorMap import TensorMap, Interpretation
from ml4cvd.tensor_maps_ecg import make_ecg_label

tmaps: Dict[str, TensorMap] = {}
tmaps["asystole"] = TensorMap(
    "asystole",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_asystole": 0, "asystole": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"asystole": {"asystole"}},
        not_found_channel="no_asystole",
    ),
)


tmaps["atrial_fibrillation"] = TensorMap(
    "atrial_fibrillation",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_atrial_fibrillation": 0, "atrial_fibrillation": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "atrial_fibrillation": {
                "atrial fib",
                "fibrillation/flutter",
                "atrial fibrillation with rapid ventricular response",
                "atrial fibrillation with controlled ventricular response",
                "atrial  fibrillation",
                "afibrillation",
                "afib",
                "atrial fibrillation with moderate ventricular response",
                "atrial fibrillation",
            },
        },
        not_found_channel="no_atrial_fibrillation",
    ),
)


tmaps["atrial_flutter"] = TensorMap(
    "atrial_flutter",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_atrial_flutter": 0, "atrial_flutter": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "atrial_flutter": {
                "atrial flutter",
                "fibrillation/flutter",
                "atrial flutter variable block",
                "tachycardia possibly flutter",
                "atrial flutter fixed block",
                "aflutter",
                "atrial flutter unspecified block",
                "probable flutter",
            },
        },
        not_found_channel="no_atrial_flutter",
    ),
)


tmaps["atrial_paced_rhythm"] = TensorMap(
    "atrial_paced_rhythm",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_atrial_paced_rhythm": 0, "atrial_paced_rhythm": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"atrial_paced_rhythm": {"atrial pacing", "atrial paced rhythm"}},
        not_found_channel="no_atrial_paced_rhythm",
    ),
)


tmaps["ectopic_atrial_bradycardia"] = TensorMap(
    "ectopic_atrial_bradycardia",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_ectopic_atrial_bradycardia": 0, "ectopic_atrial_bradycardia": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ectopic_atrial_bradycardia": {
                "ectopic atrial bradycardia",
                "low atrial bradycardia",
            },
        },
        not_found_channel="no_ectopic_atrial_bradycardia",
    ),
)


tmaps["ectopic_atrial_rhythm"] = TensorMap(
    "ectopic_atrial_rhythm",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_ectopic_atrial_rhythm": 0, "ectopic_atrial_rhythm": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ectopic_atrial_rhythm": {
                "wandering ectopic atrial rhythm",
                "ectopic atrial rhythm",
                "abnormal p vector",
                "atrial rhythm",
                "multifocal ear",
                "unusual p wave axis",
                "multiple atrial foci",
                "dual atrial foci ",
                "multifocal atrial rhythm",
                "low atrial pacer",
                "wandering atrial pacemaker",
                "wandering ear",
                "ectopicsupraventricular rhythm",
                "unifocal ear",
                "ectopic atrial rhythm ",
                "atrial arrhythmia",
                "multifocal ectopic atrial rhythm",
                "nonsinus atrial mechanism",
                "unifocal ectopic atrial rhythm",
                "p wave axis suggests atrial rather than sinus mechanism",
                "multifocal atrialrhythm",
            },
        },
        not_found_channel="no_ectopic_atrial_rhythm",
    ),
)


tmaps["ectopic_atrial_tachycardia"] = TensorMap(
    "ectopic_atrial_tachycardia",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_ectopic_atrial_tachycardia": 0, "ectopic_atrial_tachycardia": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ectopic_atrial_tachycardia": {
                "unifocal ectopic atrial tachycardia",
                "ectopic atrial tachycardia, unifocal",
                "ectopic atrial tachycardia, unspecified",
                "unspecified ectopic atrial tachycardia",
                "unifocal atrial tachycardia",
                "multifocal ectopic atrial tachycardia",
                "multifocal atrial tachycardia",
                "ectopic atrial tachycardia, multifocal",
                "ectopic atrial tachycardia",
                "wandering atrial tachycardia",
            },
        },
        not_found_channel="no_ectopic_atrial_tachycardia",
    ),
)


tmaps["narrow_qrs_tachycardia"] = TensorMap(
    "narrow_qrs_tachycardia",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_narrow_qrs_tachycardia": 0, "narrow_qrs_tachycardia": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "narrow_qrs_tachycardia": {
                "narrow complex tachycardia",
                "narrow qrs tachycardia",
                "tachycardia narrow qrs",
            },
        },
        not_found_channel="no_narrow_qrs_tachycardia",
    ),
)


tmaps["pulseless_electrical_activity"] = TensorMap(
    "pulseless_electrical_activity",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_pulseless_electrical_activity": 0,
        "pulseless_electrical_activity": 1,
    },
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "pulseless_electrical_activity": {
                "pulseless electrical activity",
                "pulseless",
            },
        },
        not_found_channel="no_pulseless_electrical_activity",
    ),
)


tmaps["retrograde_atrial_activation"] = TensorMap(
    "retrograde_atrial_activation",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_retrograde_atrial_activation": 0,
        "retrograde_atrial_activation": 1,
    },
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "retrograde_atrial_activation": {"retrograde atrial activation"},
        },
        not_found_channel="no_retrograde_atrial_activation",
    ),
)


tmaps["sinus_arrest"] = TensorMap(
    "sinus_arrest",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_sinus_arrest": 0, "sinus_arrest": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"sinus_arrest": {"sinus arrest"}},
        not_found_channel="no_sinus_arrest",
    ),
)


tmaps["sinus_pause"] = TensorMap(
    "sinus_pause",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_sinus_pause": 0, "sinus_pause": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"sinus_pause": {"sinus pauses", "sinus pause"}},
        not_found_channel="no_sinus_pause",
    ),
)


tmaps["sinus_rhythm"] = TensorMap(
    "sinus_rhythm",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_sinus_rhythm": 0, "sinus_rhythm": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "sinus_rhythm": {
                "atrial bigeminy and ventricular bigeminy",
                "rhythm has reverted to normal",
                "type i sa block",
                "1st degree sa block",
                "with occasional native sinus beats",
                "atrial bigeminal  rhythm",
                "atrialbigeminy",
                "sa block",
                "marked sinus arrhythmia",
                "normal when compared with ecg of",
                "2nd degree sa block",
                "frequent native sinus beats",
                "sa exit block",
                "rhythm is normal sinus",
                "atrial bigeminal rhythm",
                "normal ecg",
                "type i sinoatrial block",
                "sinoatrial block, type ii",
                "tracing is within normal limits",
                "type ii sa block",
                "sinus slowing",
                "atrial trigeminy",
                "rhythm remains normal sinus",
                "sinus rhythm at a rate",
                "sinus rhythm",
                "type ii sinoatrial block",
                "tracing within normal limits",
                "sinus tachycardia",
                "normal sinus rhythm",
                "sinoatrial block",
                "sinus bradycardia",
                "rhythm is now clearly sinus",
                "sinus exit block",
                "sa block, type i",
                "sinus arrhythmia",
                "sinus mechanism has replaced",
                "conducted sinus impulses",
            },
        },
        not_found_channel="no_sinus_rhythm",
    ),
)


tmaps["supraventricular_tachycardia"] = TensorMap(
    "supraventricular_tachycardia",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_supraventricular_tachycardia": 0,
        "supraventricular_tachycardia": 1,
    },
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "supraventricular_tachycardia": {
                "av reentrant tachycardia ",
                "atrial tachycardia",
                "supraventricular tachycardia",
                "atrioventricular nodal reentry tachycardia",
                "avnrt",
                "accelerated nodal rhythm",
                "avrt",
                "atrioventricular reentrant tachycardia ",
                "accelerated atrioventricular junctional rhythm",
                "junctional tachycardia",
                "accelerated atrioventricular nodal rhythm",
                "av nodal reentry tachycardia",
                "av nodal reentrant",
            },
        },
        not_found_channel="no_supraventricular_tachycardia",
    ),
)


tmaps["torsade_de_pointes"] = TensorMap(
    "torsade_de_pointes",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_torsade_de_pointes": 0, "torsade_de_pointes": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"torsade_de_pointes": {"torsade"}},
        not_found_channel="no_torsade_de_pointes",
    ),
)


tmaps["unspecified"] = TensorMap(
    "unspecified",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_unspecified": 0, "unspecified": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "unspecified": {
                "atrial arrhythmia",
                "accelerated idioventricular rhythm",
                "undetermined  rhythm",
                "undetermined rhythm",
                "uncertain rhythm",
                "technically poor tracing ",
                "rhythm unclear",
                "supraventricular rhythm",
                "rhythm uncertain",
                "atrial activity is indistinct",
            },
        },
        not_found_channel="no_unspecified",
    ),
)


tmaps["ventricular_rhythm"] = TensorMap(
    "ventricular_rhythm",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_ventricular_rhythm": 0, "ventricular_rhythm": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"ventricular_rhythm": {"accelerated idioventricular rhythm"}},
        not_found_channel="no_ventricular_rhythm",
    ),
)


tmaps["wpw"] = TensorMap(
    "wpw",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_wpw": 0, "wpw": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"wpw": {"wolffparkinsonwhite", "wolff-parkinson-white pattern"}},
        not_found_channel="no_wpw",
    ),
)


tmaps["brugada_pattern"] = TensorMap(
    "brugada_pattern",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_brugada_pattern": 0, "brugada_pattern": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"brugada_pattern": {"brugada pattern"}},
        not_found_channel="no_brugada_pattern",
    ),
)


tmaps["digitalis_effect"] = TensorMap(
    "digitalis_effect",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_digitalis_effect": 0, "digitalis_effect": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"digitalis_effect": {"digitalis effect"}},
        not_found_channel="no_digitalis_effect",
    ),
)


tmaps["early_repolarization"] = TensorMap(
    "early_repolarization",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_early_repolarization": 0, "early_repolarization": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"early_repolarization": {"early repolarization"}},
        not_found_channel="no_early_repolarization",
    ),
)


tmaps["inverted_u_waves"] = TensorMap(
    "inverted_u_waves",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_inverted_u_waves": 0, "inverted_u_waves": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"inverted_u_waves": {"inverted u waves"}},
        not_found_channel="no_inverted_u_waves",
    ),
)


tmaps["ischemia"] = TensorMap(
    "ischemia",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_ischemia": 0, "ischemia": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ischemia": {
                "consider anterior and lateral ischemia",
                "nonspecific st segment depression",
                "st segment elevation consistent with acute injury",
                "anterolateral subendocardial ischemia",
                "consider anterior ischemia",
                "st segment depression in leads",
                "st segment depressions more marked",
                "consistent with lateral ischemia",
                "st segment elevation",
                "anterior subendocardial ischemia",
                "inferior st segment elevation and q waves",
                "diffuse st segment depression",
                "marked st segment depression",
                "inferior subendocardial ischemia",
                "minor st segment depression",
                "inferoapical st segment depression",
                "consistent with ischemia",
                "anterior st segment depression",
                "antero-apical ischemia",
                "suggest anterior ischemia",
                "st segment depression in leads v4-v6",
                "suggests anterolateral ischemia",
                "suggesting anterior ischemia",
                "anterior infarct or transmural ischemia",
                "anterolateral ischemia",
                "infero- st segment depression",
                "septal ischemia",
                "st segment depression",
                "marked st segment depression in leads",
                "widespread st segment depression",
                "diffuse elevation of st segments",
                "st segment depression in anterolateral leads",
                "diffuse scooped st segment depression",
                "apical subendocardial ischemia",
                "st segment elevation in leads",
                "possible anterior wall ischemia",
                "st segment depression is more marked in leads",
                "diffuse st segment elevation",
                "apical st depression",
                "anterolateral st segment depression",
                "subendocardial ischemia",
                "consistent with subendocardial ischemia",
                "st depression",
                "st elevation",
            },
        },
        not_found_channel="no_ischemia",
    ),
)


tmaps["metabolic_or_drug_effect"] = TensorMap(
    "metabolic_or_drug_effect",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_metabolic_or_drug_effect": 0, "metabolic_or_drug_effect": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"metabolic_or_drug_effect": {"metabolic or drug effect"}},
        not_found_channel="no_metabolic_or_drug_effect",
    ),
)


tmaps["osborn_wave"] = TensorMap(
    "osborn_wave",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_osborn_wave": 0, "osborn_wave": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"osborn_wave": {"osborn wave"}},
        not_found_channel="no_osborn_wave",
    ),
)


tmaps["pericarditis"] = TensorMap(
    "pericarditis",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_pericarditis": 0, "pericarditis": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"pericarditis": {"pericarditis"}},
        not_found_channel="no_pericarditis",
    ),
)


tmaps["prominent_u_waves"] = TensorMap(
    "prominent_u_waves",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_prominent_u_waves": 0, "prominent_u_waves": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"prominent_u_waves": {"prominent u waves"}},
        not_found_channel="no_prominent_u_waves",
    ),
)


tmaps["st_abnormality"] = TensorMap(
    "st_abnormality",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_st_abnormality": 0, "st_abnormality": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "st_abnormality": {
                "nonspecific st segment depression",
                "st segment changes",
                "st segment elevation consistent with acute injury",
                "st segment depression in leads",
                "st segment depressions more marked",
                "st segment elevation",
                "inferior st segment elevation and q waves",
                "diffuse st segment depression",
                "marked st segment depression",
                "minor st segment depression",
                "inferoapical st segment depression",
                "anterior st segment depression",
                "st segment depression in leads v4-v6",
                "abnormal st segment changes",
                "infero- st segment depression",
                "st segment depression",
                "marked st segment depression in leads",
                "widespread st segment depression",
                "diffuse elevation of st segments",
                "st segment depression in anterolateral leads",
                "diffuse scooped st segment depression",
                "st segment elevation in leads",
                "nonspecific st segment and t wave abnormalities",
                "st segment depression is more marked in leads",
                "diffuse st segment elevation",
                "apical st depression",
                "anterolateral st segment depression",
                "st depression",
                "nonspecific st segment",
                "st segment abnormality",
                "st elevation",
            },
        },
        not_found_channel="no_st_abnormality",
    ),
)


tmaps["st_or_t_change_due_to_ventricular_hypertrophy"] = TensorMap(
    "st_or_t_change_due_to_ventricular_hypertrophy",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_st_or_t_change_due_to_ventricular_hypertrophy": 0,
        "st_or_t_change_due_to_ventricular_hypertrophy": 1,
    },
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "st_or_t_change_due_to_ventricular_hypertrophy": {
                "st or t change due to ventricular hypertrophy",
            },
        },
        not_found_channel="no_st_or_t_change_due_to_ventricular_hypertrophy",
    ),
)


tmaps["t_wave_abnormality"] = TensorMap(
    "t_wave_abnormality",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_t_wave_abnormality": 0, "t_wave_abnormality": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "t_wave_abnormality": {
                "t wave inveions",
                "diffuse nonspecific st segment and t wave abnormalities",
                "t waves are inverted in leads",
                "t wave inversions",
                "nonspecific t wave abnormali",
                "possible st segment and t wave abn",
                "t wave flattening",
                "t waves are upright in leads",
                "tall t waves in precordial leads",
                "(nonspecific st segment).*(t wave abnormalities)",
                "t wave inversion in leads",
                "t wave inversion",
                "t wave inver",
                "upright t waves",
                "recent diffuse t wave flattening",
                "nonspecific st segment and t wave abnormalities",
                "t waves are slightly more inverted in leads",
                "t wave abnormalities",
                "t waves are lower or inverted in leads",
                "t wave changes",
            },
        },
        not_found_channel="no_t_wave_abnormality",
    ),
)


tmaps["tu_fusion"] = TensorMap(
    "tu_fusion",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_tu_fusion": 0, "tu_fusion": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"tu_fusion": {"tu fusion"}},
        not_found_channel="no_tu_fusion",
    ),
)


tmaps["fascicular_rhythm"] = TensorMap(
    "fascicular_rhythm",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_fascicular_rhythm": 0, "fascicular_rhythm": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"fascicular_rhythm": {"fascicular rhythm"}},
        not_found_channel="no_fascicular_rhythm",
    ),
)


tmaps["fusion_complexes"] = TensorMap(
    "fusion_complexes",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_fusion_complexes": 0, "fusion_complexes": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"fusion_complexes": {"fusion complexes", "fusion beats"}},
        not_found_channel="no_fusion_complexes",
    ),
)


tmaps["idioventricular_rhythm"] = TensorMap(
    "idioventricular_rhythm",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_idioventricular_rhythm": 0, "idioventricular_rhythm": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"idioventricular_rhythm": {"idioventricular rhythm"}},
        not_found_channel="no_idioventricular_rhythm",
    ),
)


tmaps["junctional_rhythm"] = TensorMap(
    "junctional_rhythm",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_junctional_rhythm": 0, "junctional_rhythm": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"junctional_rhythm": {"junctional rhythm"}},
        not_found_channel="no_junctional_rhythm",
    ),
)


tmaps["parasystole"] = TensorMap(
    "parasystole",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_parasystole": 0, "parasystole": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"parasystole": {"parasystole"}},
        not_found_channel="no_parasystole",
    ),
)


tmaps["ventricular_fibrillation"] = TensorMap(
    "ventricular_fibrillation",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_ventricular_fibrillation": 0, "ventricular_fibrillation": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"ventricular_fibrillation": {"ventricular fibrillation"}},
        not_found_channel="no_ventricular_fibrillation",
    ),
)


tmaps["ventricular_tachycardia"] = TensorMap(
    "ventricular_tachycardia",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_ventricular_tachycardia": 0, "ventricular_tachycardia": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ventricular_tachycardia": {
                " ventricular tachy",
                "\\w*(?<!supra)(ventricular tachycardia)",
            },
        },
        not_found_channel="no_ventricular_tachycardia",
    ),
)


tmaps["wide_qrs_rhythm"] = TensorMap(
    "wide_qrs_rhythm",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_wide_qrs_rhythm": 0, "wide_qrs_rhythm": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"wide_qrs_rhythm": {"wide qrs rhythm"}},
        not_found_channel="no_wide_qrs_rhythm",
    ),
)


tmaps["first_degree_av_block"] = TensorMap(
    "first_degree_av_block",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_first_degree_av_block": 0, "first_degree_av_block": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "first_degree_av_block": {
                "1st degree atrioventricular  block",
                "first degree atrioventricular block",
                "first degree atrioventricular  block",
                "first degree atrioventricular block ",
                "first degree atrioventricular",
                "first degree avb",
                "first degree av block",
            },
        },
        not_found_channel="no_first_degree_av_block",
    ),
)


tmaps["aberrant_conduction_of_supraventricular_beats"] = TensorMap(
    "aberrant_conduction_of_supraventricular_beats",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_aberrant_conduction_of_supraventricular_beats": 0,
        "aberrant_conduction_of_supraventricular_beats": 1,
    },
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "aberrant_conduction_of_supraventricular_beats": {
                "aberrant conduction",
                "aberrant conduction of supraventricular beats",
            },
        },
        not_found_channel="no_aberrant_conduction_of_supraventricular_beats",
    ),
)


tmaps["bundle_branch_block"] = TensorMap(
    "bundle_branch_block",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_bundle_branch_block": 0, "bundle_branch_block": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"bundle_branch_block": {"bundle branch block", "bbb"}},
        not_found_channel="no_bundle_branch_block",
    ),
)


tmaps["crista_pattern"] = TensorMap(
    "crista_pattern",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_crista_pattern": 0, "crista_pattern": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"crista_pattern": {"crista pattern"}},
        not_found_channel="no_crista_pattern",
    ),
)


tmaps["epsilon_wave"] = TensorMap(
    "epsilon_wave",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_epsilon_wave": 0, "epsilon_wave": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"epsilon_wave": {"epsilon wave"}},
        not_found_channel="no_epsilon_wave",
    ),
)


tmaps["incomplete_left_bundle_branch_block"] = TensorMap(
    "incomplete_left_bundle_branch_block",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_incomplete_left_bundle_branch_block": 0,
        "incomplete_left_bundle_branch_block": 1,
    },
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "incomplete_left_bundle_branch_block": {
                "incomplete left bundle branch block",
            },
        },
        not_found_channel="no_incomplete_left_bundle_branch_block",
    ),
)


tmaps["incomplete_right_bundle_branch_block"] = TensorMap(
    "incomplete_right_bundle_branch_block",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_incomplete_right_bundle_branch_block": 0,
        "incomplete_right_bundle_branch_block": 1,
    },
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "incomplete_right_bundle_branch_block": {
                "incomplete right bundle branch block",
            },
        },
        not_found_channel="no_incomplete_right_bundle_branch_block",
    ),
)


tmaps["intraventricular_conduction_delay"] = TensorMap(
    "intraventricular_conduction_delay",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_intraventricular_conduction_delay": 0,
        "intraventricular_conduction_delay": 1,
    },
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "intraventricular_conduction_delay": {
                "intraventricular conduction delay",
                "intraventricular conduction defect",
            },
        },
        not_found_channel="no_intraventricular_conduction_delay",
    ),
)


tmaps["left_anterior_fascicular_block"] = TensorMap(
    "left_anterior_fascicular_block",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_left_anterior_fascicular_block": 0,
        "left_anterior_fascicular_block": 1,
    },
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "left_anterior_fascicular_block": {
                "left anterior fascicular block",
                "left anterior hemiblock",
            },
        },
        not_found_channel="no_left_anterior_fascicular_block",
    ),
)


tmaps["left_atrial_conduction_abnormality"] = TensorMap(
    "left_atrial_conduction_abnormality",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_left_atrial_conduction_abnormality": 0,
        "left_atrial_conduction_abnormality": 1,
    },
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "left_atrial_conduction_abnormality": {
                "left atrial conduction abnormality",
            },
        },
        not_found_channel="no_left_atrial_conduction_abnormality",
    ),
)


tmaps["left_bundle_branch_block"] = TensorMap(
    "left_bundle_branch_block",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_left_bundle_branch_block": 0, "left_bundle_branch_block": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"left_bundle_branch_block": {"left bundle branch block"}},
        not_found_channel="no_left_bundle_branch_block",
    ),
)


tmaps["left_posterior_fascicular_block"] = TensorMap(
    "left_posterior_fascicular_block",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_left_posterior_fascicular_block": 0,
        "left_posterior_fascicular_block": 1,
    },
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "left_posterior_fascicular_block": {
                "left posterior hemiblock",
                "left posterior fascicular block",
            },
        },
        not_found_channel="no_left_posterior_fascicular_block",
    ),
)


tmaps["nonspecific_ivcd"] = TensorMap(
    "nonspecific_ivcd",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_nonspecific_ivcd": 0, "nonspecific_ivcd": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"nonspecific_ivcd": {"nonspecific ivcd"}},
        not_found_channel="no_nonspecific_ivcd",
    ),
)


tmaps["right_atrial_conduction_abnormality"] = TensorMap(
    "right_atrial_conduction_abnormality",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_right_atrial_conduction_abnormality": 0,
        "right_atrial_conduction_abnormality": 1,
    },
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "right_atrial_conduction_abnormality": {
                "right atrial conduction abnormality",
            },
        },
        not_found_channel="no_right_atrial_conduction_abnormality",
    ),
)


tmaps["right_bundle_branch_block"] = TensorMap(
    "right_bundle_branch_block",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_right_bundle_branch_block": 0, "right_bundle_branch_block": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"right_bundle_branch_block": {"right bundle branch block"}},
        not_found_channel="no_right_bundle_branch_block",
    ),
)


tmaps["ventricular_preexcitation"] = TensorMap(
    "ventricular_preexcitation",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_ventricular_preexcitation": 0, "ventricular_preexcitation": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"ventricular_preexcitation": {"ventricular preexcitation"}},
        not_found_channel="no_ventricular_preexcitation",
    ),
)


tmaps["atrioventricular_dissociation"] = TensorMap(
    "atrioventricular_dissociation",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_atrioventricular_dissociation": 0,
        "atrioventricular_dissociation": 1,
    },
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "atrioventricular_dissociation": {"atrioventricular dissociation"},
        },
        not_found_channel="no_atrioventricular_dissociation",
    ),
)


tmaps["av_dissociation"] = TensorMap(
    "av_dissociation",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_av_dissociation": 0, "av_dissociation": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"av_dissociation": {"av dissociation"}},
        not_found_channel="no_av_dissociation",
    ),
)


tmaps["_2_to_1_av_block"] = TensorMap(
    "_2_to_1_av_block",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no__2_to_1_av_block": 0, "_2_to_1_av_block": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "_2_to_1_av_block": {
                "2 to 1 av block",
                "2:1 av block",
                "2:1 atrioventricular block",
                "2:1 block",
                "2 to 1 atrioventricular block",
            },
        },
        not_found_channel="no__2_to_1_av_block",
    ),
)


tmaps["_4_to_1_av_block"] = TensorMap(
    "_4_to_1_av_block",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no__4_to_1_av_block": 0, "_4_to_1_av_block": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"_4_to_1_av_block": {"4:1atrioventricular conduction"}},
        not_found_channel="no__4_to_1_av_block",
    ),
)


tmaps["av_dissociation"] = TensorMap(
    "av_dissociation",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_av_dissociation": 0, "av_dissociation": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "av_dissociation": {"atrioventricular dissociation", "av dissociation"},
        },
        not_found_channel="no_av_dissociation",
    ),
)


tmaps["mobitz_type_i_second_degree_av_block_"] = TensorMap(
    "mobitz_type_i_second_degree_av_block_",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_mobitz_type_i_second_degree_av_block_": 0,
        "mobitz_type_i_second_degree_av_block_": 1,
    },
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "mobitz_type_i_second_degree_av_block_": {
                "mobitz 1 block",
                "mobitz i",
                "mobitz type 1",
                "second degree ",
                "fixed block",
                "second degree type 1",
                "wenckebach",
            },
        },
        not_found_channel="no_mobitz_type_i_second_degree_av_block_",
    ),
)


tmaps["mobitz_type_ii_second_degree_av_block"] = TensorMap(
    "mobitz_type_ii_second_degree_av_block",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_mobitz_type_ii_second_degree_av_block": 0,
        "mobitz_type_ii_second_degree_av_block": 1,
    },
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "mobitz_type_ii_second_degree_av_block": {
                "hay block",
                "second degree type 2",
                "mobitz ii",
                "2nd degree sa block",
            },
        },
        not_found_channel="no_mobitz_type_ii_second_degree_av_block",
    ),
)


tmaps["third_degree_avb"] = TensorMap(
    "third_degree_avb",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_third_degree_avb": 0, "third_degree_avb": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "third_degree_avb": {
                "complete heart block",
                "3rd degree av block",
                "third degree atrioventricular block",
                "3rd degree atrioventricular block",
                "third degree av block",
            },
        },
        not_found_channel="no_third_degree_avb",
    ),
)


tmaps["unspecified_avb"] = TensorMap(
    "unspecified_avb",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_unspecified_avb": 0, "unspecified_avb": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "unspecified_avb": {
                "heart block",
                "high degree of block",
                "atrioventricular block",
                "av block",
                "heartblock",
                "high grade atrioventricular block",
            },
        },
        not_found_channel="no_unspecified_avb",
    ),
)


tmaps["variable_avb"] = TensorMap(
    "variable_avb",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_variable_avb": 0, "variable_avb": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"variable_avb": {"variable block", "varying degree of block"}},
        not_found_channel="no_variable_avb",
    ),
)


tmaps["atrial_premature_complexes"] = TensorMap(
    "atrial_premature_complexes",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_atrial_premature_complexes": 0, "atrial_premature_complexes": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "atrial_premature_complexes": {
                "isolated premature atrial contractions",
                "atrial ectopy has decreased",
                "premature atrial complexes",
                "atrial premature complexes",
                "atrial trigeminy",
                "ectopic atrial complexes",
                "premature atrial co",
                "atrial premature beat",
                "atrial ectopy",
                "atrial bigeminy",
            },
        },
        not_found_channel="no_atrial_premature_complexes",
    ),
)


tmaps["ectopy"] = TensorMap(
    "ectopy",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_ectopy": 0, "ectopy": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ectopy": {
                "ectopy is new",
                "ectopy more pronounced",
                "other than the ectopy",
                "ectopy has appeared",
                "ectopy have increased",
                "return of ectopy",
            },
        },
        not_found_channel="no_ectopy",
    ),
)


tmaps["junctional_premature_complexes"] = TensorMap(
    "junctional_premature_complexes",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_junctional_premature_complexes": 0,
        "junctional_premature_complexes": 1,
    },
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "junctional_premature_complexes": {
                "junctional premature complexes",
                "junctional premature beats",
            },
        },
        not_found_channel="no_junctional_premature_complexes",
    ),
)


tmaps["no_ectopy"] = TensorMap(
    "no_ectopy",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_no_ectopy": 0, "no_ectopy": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "no_ectopy": {
                "ectopy is no longer seen",
                "no ectopy",
                "no longer any ectopy",
                "ectopy has resolved",
                "ectopy is gone",
                "atrial ectopy gone",
                "ectopy has disappear",
            },
        },
        not_found_channel="no_no_ectopy",
    ),
)


tmaps["premature_supraventricular_complexes"] = TensorMap(
    "premature_supraventricular_complexes",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_premature_supraventricular_complexes": 0,
        "premature_supraventricular_complexes": 1,
    },
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "premature_supraventricular_complexes": {
                "premature supraventricular complexes",
            },
        },
        not_found_channel="no_premature_supraventricular_complexes",
    ),
)


tmaps["ventricular_premature_complexes"] = TensorMap(
    "ventricular_premature_complexes",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_ventricular_premature_complexes": 0,
        "ventricular_premature_complexes": 1,
    },
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ventricular_premature_complexes": {
                "premature ventricular beat",
                "ventricular ectopy",
                "ventricular premature beat",
                "ventricular trigeminy",
                "isolated premature ventricular contractions",
                "frequent premature ventricular or aberrantly conducted complexes ",
                "premature ventricular compl",
                "premature ventricular complexes",
                "one premature ventricularbeat",
                "ventricular bigeminy",
                "ventriculaar ectopy is now present",
                "premature ventricular and fusion complexes",
                "premature ventricular complexe",
                "ventricular premature complexes",
                "occasional premature ventricular complexes ",
            },
        },
        not_found_channel="no_ventricular_premature_complexes",
    ),
)


tmaps["mi"] = TensorMap(
    "mi",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_mi": 0, "mi": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "mi": {
                "extensive anterolateral myocardial infarction",
                "known true posterior myocardial infarction",
                "normal counterclockwise rotation versus old true posterior myocardial infarction",
                "old inferior and anterior myocardial infarctions",
                "myocardial infarction of indeterminate age",
                "probable apicolateral myocardial infarction",
                "myocardial infarction compared with the last previous ",
                "myocardial infarction when compared with ecg of",
                "extensive myocardial infarction of indeterminate age ",
                "anteroseptal infarct of indeterminate age",
                "old anterior infarct",
                "raises possibility of septal infarct",
                "suggestive of old true posterior myocardial infarction st abnormality",
                "anterolateral myocardial infarction appears recent",
                "subendocardial infarct",
                "infero and apicolateral myocardial infarction",
                "acute infarct",
                "evolving myocardial infarction",
                "old inferior posterolateral myocardial infarction",
                "anterolateral myocardial infarction",
                "age indeterminate old inferior wall myocardial infarction",
                "old infero-postero-lateral myocardial infarction",
                "old inferoapical myocardial infarction",
                "acute anterior wall myocardial infarction",
                "antero-apical and lateral myocardial infarction evolving",
                "(consistent with).*(true posterior).*(myocardial infarction)",
                "old infero-posterior lateral myocardial infarction",
                "old anterior myocardial infarction",
                "myocardial infarction cannot rule out",
                "subendocardial ischemia myocardial infarction",
                "anterior myocardial infarction of indeterminate age",
                "possible old septal myocardial infarction",
                "anteroseptal and lateral myocardial infarction",
                "counterclockwise rotation versus old true posterior myocardial infarction",
                "acute myocardial infarction",
                "consistent with anteroseptal infarct",
                "inferior myocardial infarction of indeterminate age",
                "acuteanterior myocardial infarction",
                "old infero-posterior and apical myocardial infarction",
                "rule out interim myocardial infarction",
                "apical myocardial infarction",
                "ongoing ischemia versus myocardial infarction",
                "possible acute inferior myocardial infarction",
                "lateral myocardial infarction of indeterminate age",
                "inferior myocardial infarction , age undetermined",
                "(myocardial infarction versus).*(counterclockwise rotation)",
                "old high lateral myocardial infarction",
                "(counterclockwise rotation).*(true posterior)",
                "evolving counterclockwise rotation versus true posterior myocardial infarction",
                "old inferior wall myocardial infarction",
                "acute myocardial infarction of indeterminate age",
                "normal counterclockwise rotation versus true posterior myocardial infarction",
                "subendocardial ischemia or myocardial infarction",
                "myocardial infarction",
                "subendocardial ischemia subendocardial myocardial inf",
                "cannot rule out anterior wall ischemia myocardial infarction",
                "consistent with anterior myocardial infarction of indeterminate age",
                "old apicolateral myocardial infarction",
                "old inferolateral myocardial infarction",
                "normal left axis deviation counterclockwise rotation versus old true posterior myocardial infarction",
                "extensive anterior infarct",
                "inferior wall myocardial infarction of indeterminate age",
                "old anteroseptal myocardial infarction",
                "acute myocardial infarction in evolution",
                "cannot rule out anterior infarct , age undetermined",
                "old inferoposterior myocardial infarction",
                "myocardial infarction compared with the last previous tracing ",
                "post myocardial infarction , of indeterminate age",
                "counterclockwise rotation consistent with post myocardial infarction",
                "posterior wall myocardial infarction",
                "(possible old).*(true posterior).*(myocardial infarction)",
                "possible acute myocardial infarction",
                "recent myocardial infarction",
                "infero-apical myocardial infarction",
                "old inferior myocardial infarction",
                "anterolateral infarct of indeterminate age ",
                "inferoapical myocardial infarction of indeterminate age",
                "possible anteroseptal myocardial infarction ,of uncertain age",
                "inferior myocardial infarction",
                "lateral myocardial infarction - of indeterminate age",
                "myocardial infarction nonspecific st segment",
                "cannot rule out anteroseptal infarct",
                "possible anterolateral myocardial infarction",
                "evolving anterior infarct",
                "old inferior anterior myocardial infarctions",
                "cannot rule out true posterior myocardial infarction",
                "myocardial infarction old high lateral",
                "myocardial infarction indeterminate",
                "old infero-posterior myocardial infarction",
                "inferior myocardial infarction of indeterminate",
                "borderline anterolateral myocardial infarction",
                "possible anteroseptal myocardial infarction",
                "myocardial infarction extension",
                "possible septal myocardial infarction",
                "suggestive of old true posterior myocardial infarction",
                "old myocardial infarction",
                "transmural ischemia myocardial infarction",
                "normal counterclockwise rotation versusold true posterior myocardial infarction",
                "evolving inferior wall myocardial infarction",
                "inferior myocardial infarction , possibly acute",
                "evolution of myocardial infarction",
                "old lateral myocardial infarction",
                "old anterolateral infarct",
                "possible old lateral myocardial infarction",
                "old anterior wall myocardial infarction",
                "myocardial infarction versus pericarditis",
                "consistent with ischemia myocardial infarction",
                "new incomplete posterior lateral myocardial infarction",
                "anterolateral myocardial infarction , possibly recent  ",
                "antero-apical ischemia versus myocardial infarction",
                "inferolateral myocardial infarction",
                "possible inferior myocardial infarction",
                "subendocardial myocardial infarction",
                "possible true posterior myocardial infarction",
                "cannot rule out true posterior myocardial infarction versus counterclockwise rotation",
                "lateral wall myocardial infarction",
                "block inferior myocardial infarction",
                "anterior myocardial infarction",
                "counterclockwise rotation versus true posterior myocardial infarction",
                "septal infarct",
                "myocardial infarction possible when compared",
                "(true posterior).*(myocardial infarction)",
                "cannot rule out inferoposterior myoca",
                "concurrent ischemia myocardial infarction",
                "posterior myocardial infarction",
                "possible myocardial infarction",
                "old anteroseptal infarct",
                "old anterolateral myocardial infarction",
                "anterior infarct of indeterminate age",
                "acute anterior infarct",
                "anteroseptal myocardial infarction",
                "old true posterior myocardial infarction",
                "old posterolateral myocardial infarction",
                "apical myocardial infarction of indeterminate age",
                "infero-apical myocardial infarction of indeterminate age",
                "myocardial infarction pattern",
                "true posterior myocardial infarction of indeterminate age",
            },
        },
        not_found_channel="no_mi",
    ),
)


tmaps["abnormal_p_wave_axis"] = TensorMap(
    "abnormal_p_wave_axis",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_abnormal_p_wave_axis": 0, "abnormal_p_wave_axis": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"abnormal_p_wave_axis": {"abnormal p wave axis"}},
        not_found_channel="no_abnormal_p_wave_axis",
    ),
)


tmaps["electrical_alternans"] = TensorMap(
    "electrical_alternans",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_electrical_alternans": 0, "electrical_alternans": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"electrical_alternans": {"electrical alternans"}},
        not_found_channel="no_electrical_alternans",
    ),
)


tmaps["ignore"] = TensorMap(
    "ignore",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_ignore": 0, "ignore": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"ignore": {"r wave progression has improved"}},
        not_found_channel="no_ignore",
    ),
)


tmaps["indeterminate_axis"] = TensorMap(
    "indeterminate_axis",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_indeterminate_axis": 0, "indeterminate_axis": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "indeterminate_axis": {
                "northwest axis",
                "indeterminate axis",
                "indeterminate qrs axis",
            },
        },
        not_found_channel="no_indeterminate_axis",
    ),
)


tmaps["left_axis_deviation"] = TensorMap(
    "left_axis_deviation",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_left_axis_deviation": 0, "left_axis_deviation": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "left_axis_deviation": {
                "left axis deviation",
                "axis shifted left",
                "leftward axis",
            },
        },
        not_found_channel="no_left_axis_deviation",
    ),
)


tmaps["low_voltage"] = TensorMap(
    "low_voltage",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_low_voltage": 0, "low_voltage": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"low_voltage": {"low voltage"}},
        not_found_channel="no_low_voltage",
    ),
)


tmaps["normal_axis"] = TensorMap(
    "normal_axis",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_normal_axis": 0, "normal_axis": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"normal_axis": {"normal axis"}},
        not_found_channel="no_normal_axis",
    ),
)


tmaps["poor_r_wave_progression"] = TensorMap(
    "poor_r_wave_progression",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_poor_r_wave_progression": 0, "poor_r_wave_progression": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "poor_r_wave_progression": {
                "abnormal precordial r wave progression or poor r wave progression",
                "slowprecordial r wave progression",
                "slow precordial r wave progression",
                "unusual r wave progression",
                "poor precordial r wave progression",
                "early r wave progression",
                "abnormal precordial r wave progression",
                "poor r wave progression",
            },
        },
        not_found_channel="no_poor_r_wave_progression",
    ),
)


tmaps["reverse_r_wave_progression"] = TensorMap(
    "reverse_r_wave_progression",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_reverse_r_wave_progression": 0, "reverse_r_wave_progression": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "reverse_r_wave_progression": {
                "reverse r wave progression",
                "reversed r wave progression",
            },
        },
        not_found_channel="no_reverse_r_wave_progression",
    ),
)


tmaps["right_axis_deviation"] = TensorMap(
    "right_axis_deviation",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_right_axis_deviation": 0, "right_axis_deviation": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "right_axis_deviation": {
                "right superior axis deviation",
                "right axis deviation",
                "rightward axis",
                "axis shifted right",
            },
        },
        not_found_channel="no_right_axis_deviation",
    ),
)


tmaps["right_superior_axis"] = TensorMap(
    "right_superior_axis",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_right_superior_axis": 0, "right_superior_axis": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"right_superior_axis": {"right superior axis"}},
        not_found_channel="no_right_superior_axis",
    ),
)


tmaps["bae"] = TensorMap(
    "bae",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_bae": 0, "bae": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"bae": {"biatrial enlargement"}},
        not_found_channel="no_bae",
    ),
)


tmaps["lae"] = TensorMap(
    "lae",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_lae": 0, "lae": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "lae": {
                "right atrial enla",
                "left atrial enlargement",
                "combined atrial enlargement",
                "left atrial enlarge",
                "biatrial hypertrophy",
            },
        },
        not_found_channel="no_lae",
    ),
)


tmaps["lvh"] = TensorMap(
    "lvh",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_lvh": 0, "lvh": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "lvh": {
                "biventricular hypertrophy",
                "left ventricular hypertr",
                "left ventricular hypertro",
                "combined ventricular hypertrophy",
                "biventriclar hypertrophy",
                "left ventricular hypertrophy",
                "leftventricular hypertrophy",
                "left ventricular hypertroph",
            },
        },
        not_found_channel="no_lvh",
    ),
)


tmaps["rae"] = TensorMap(
    "rae",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_rae": 0, "rae": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "rae": {
                "right atrial enlargement",
                "combined atrial enlargement",
                "biatrial hypertrophy",
            },
        },
        not_found_channel="no_rae",
    ),
)


tmaps["rvh"] = TensorMap(
    "rvh",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_rvh": 0, "rvh": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "rvh": {
                "rightventricular hypertrophy",
                "right ventricular enlargement",
                "biventricular hypertrophy",
                "biventriclar hypertrophy",
                "combined ventricular hypertrophy",
                "right ventricular hypertrophy",
            },
        },
        not_found_channel="no_rvh",
    ),
)


tmaps["sh"] = TensorMap(
    "sh",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_sh": 0, "sh": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"sh": {"septal lipomatous hypertrophy", "septal hypertrophy"}},
        not_found_channel="no_sh",
    ),
)


tmaps["pacemaker"] = TensorMap(
    "pacemaker",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_pacemaker": 0, "pacemaker": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "pacemaker": {
                "atrial triggered ventricular pacing",
                "unipolar right ventricular  pacing",
                "failure to pace ventricular",
                "atrially triggered v paced",
                "sequential pacing",
                "biventricular-paced complexes",
                "failure to inhibit ventricular",
                "atrial-paced complexes ",
                "v-paced rhythm",
                "ventricular demand pacing",
                "electronic pacemaker",
                "ventricular-paced rhythm",
                "biventricular-paced rhythm",
                "av dual-paced rhythm",
                "atrial-sensed ventricular-paced rhythm",
                "shows dual chamber pacing",
                "a triggered v-paced rhythm",
                "atrial-paced rhythm",
                "atrial-sensed ventricular-paced complexes",
                "ventricular paced",
                "failure to pace atrial",
                "av dual-paced complexes",
                "demand v-pacing",
                "demand ventricular pacemaker",
                "ventricular-paced complexes",
                "failure to inhibit atrial",
                "ventricular pacing has replaced av pacing",
                "ventricular pacing",
                "dual chamber pacing",
                "failure to capture atrial",
                "failure to capture ventricular",
                "competitive av pacing",
                "v-paced beats",
                "v-paced",
            },
        },
        not_found_channel="no_pacemaker",
    ),
)


tmaps["abnormal_ecg"] = TensorMap(
    "abnormal_ecg",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_abnormal_ecg": 0, "abnormal_ecg": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"abnormal_ecg": {"abnormal ecg", "abnormal"}},
        not_found_channel="no_abnormal_ecg",
    ),
)


tmaps["normal_sinus_rhythm"] = TensorMap(
    "normal_sinus_rhythm",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_normal_sinus_rhythm": 0, "normal_sinus_rhythm": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "normal_sinus_rhythm": {
                "normal tracing",
                "normal ecg",
                "normal sinus rhythm",
                "tracing is within normal limits",
                "sinus rhythm",
                "sinus tachycardia",
            },
        },
        not_found_channel="no_normal_sinus_rhythm",
    ),
)


tmaps["uninterpretable_ecg"] = TensorMap(
    "uninterpretable_ecg",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_uninterpretable_ecg": 0, "uninterpretable_ecg": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"uninterpretable_ecg": {"uninterpretable ecg"}},
        not_found_channel="no_uninterpretable_ecg",
    ),
)
