from ml4cvd.TensorMap import TensorMap
from ml4cvd.tensor_maps_by_hand import TMAPS
from ml4cvd.tensor_from_file import make_partners_ecg_reads

TMAPS['supranodal_rhythms'] = TensorMap('supranodal_rhythms', group='categorical', channel_map={'svt': 0, 'atrial_flutter': 1, 'ectopic_atrial_tachycardia': 2, 'atrial_fibrillation': 3, 'sinus_rhythm': 4, 'retrograde_atrial_activation': 5, 'ectopic_atrial_rhythm': 6, 'narrow_qrs_tachycardia': 7, 'unspecified': 8, }, tensor_from_file=make_partners_ecg_reads({'sinus_rhythm': ['marked sinus arrhythmia', 'sinus arrhythmia', 'normal ecg', 'atrial bigeminal rhythm', 'atrial bigeminy and ventricular bigeminy', 'atrial trigeminy', 'atrialbigeminy', 'normal sinus rhythm', 'normal when compared with ecg of', 'rhythm has reverted to normal', 'rhythm is now clearly sinus', 'sinus bradycardia', 'sinus rhythm', 'sinus rhythm at a rate', 'sinus tachycardia', 'tracing within normal limits', 'tracing is within normal limits', 'tracing within normal limits', 'sinoatrial block', 'sa block', 'sinoatrial block, type ii', 'sa block, type i', 'type i sinoatrial block', 'type i sa block', 'sinoatrial block, type ii', 'sa block, type i', 'type ii sinoatrial block', 'type ii sa block', 'sinus pause', 'sinus arrest', 'sinus slowing', 'with occasional native sinus beats', 'frequent native sinus beats', 'conducted sinus impulses', 'sinus mechanism has replaced', 'rhythm is normal sinus', 'rhythm remains normal sinus'], 'retrograde_atrial_activation': ['retrograde atrial activation'], 'ectopic_atrial_rhythm': ['unifocal atrial tachycardia', 'unifocal ectopic atrial rhythm', 'unifocal ear', 'multifocal atrial tachycardia', 'multifocal ectopic atrial rhythm', 'multifocal ear', 'wandering atrial tachycardia', 'wandering ectopic atrial rhythm', 'wandering ear', 'wandering atrial pacemaker', 'atrial rhythm', 'ectopic atrial rhythm', 'ear', 'dual atrial foci ', 'ectopic atrial bradycardia', 'multiple atrial foci', 'abnormal p vector', 'nonsinus atrial mechanism', 'p wave axis suggests atrial rather than sinus mechanism', 'low atrial bradycardia', 'multifocal atrial rhythm', 'multifocal atrialrhythm', 'unusual p wave axis', 'low atrial pacer'], 'atrial_fibrillation': ['af', 'afib', 'atrial fibrillation', 'atrial fibrillation with controlled ventricular response', 'atrial fibrillation with moderate ventricular response', 'atrial fibrillation with rapid ventricular response', 'atrial fibrillation with rvr', 'afib with rvr'], 'atrial_flutter': ['fibrillation/flutter', 'aflutter', 'atrial flutter', 'probable flutter', 'atrial flutter fixed block', 'atrial flutter variable block', 'atrial flutter unspecified block', 'tachycardia possibly flutter'], 'ectopic_atrial_tachycardia': ['ectopic atrial tachycardia', 'ectopic atrial tachycardia, unspecified', 'unspecified ectopic atrial tachycardia', 'ectopic atrial tachycardia, unifocal', 'unifocal ectopic atrial tachycardia', 'ectopic atrial tachycardia, multifocal', 'multifocal ectopic atrial tachycardia'], 'svt': ['sinus arrhythmia accelerated atrioventricular junctional rhythm', 'supraventricular tachycardia', 'accelerated atrioventricular nodal rhythm', 'accelerated nodal rhythm', 'atrial tachycardia', 'av nodal reentry tachycardia', 'atrioventricular nodal reentry tachycardia', 'avnrt', 'atrioventricular reentrant tachycardia ', 'av reentrant tachycardia ', 'avrt'], 'narrow_qrs_tachycardia': ['narrow qrs tachycardia', 'narrow qrs tachycardia', 'tachycardia narrow qrs', 'narrow complex tachycardia'], 'unspecified': ['junctional tachycardia', 'atrial arrhythmia', 'technically poor tracing ', 'accelerated idioventricular rhythm', 'atrial activity is indistinct', 'rhythm uncertain', 'rhythm unclear', 'uncertain rhythm', 'undetermined rhythm', 'supraventricular rhythm']})) 

TMAPS['sinus_rhythm'] = TensorMap('sinus_rhythm', group='categorical', channel_map={'sinus_arrhythmia': 0, 'unspecified': 1}, tensor_from_file=make_partners_ecg_reads({'sinus_arrhythmia': ['marked sinus arrhythmia', 'marked sinus arrhythmia', 'sinus arrhythmia', 'sinus arrhythmia']})) 

TMAPS['ectopic_atrial_rhythm'] = TensorMap('ectopic_atrial_rhythm', group='categorical', channel_map={'multifocal': 0, 'unifocal': 1, 'wandering': 2, 'unspecified': 3, }, tensor_from_file=make_partners_ecg_reads({'unifocal': ['unifocal atrial tachycardia', 'unifocal atrial tachycardia', 'unifocal ectopic atrial rhythm', 'unifocal ectopic atrial rhythm', 'unifocal ear', 'unifocal ear', 'unusual p wave axis', 'unusual p wave axis', 'low atrial pacer', 'low atrial pacer'], 'multifocal': ['multifocal atrial tachycardia', 'multifocal atrial tachycardia', 'multifocal ectopic atrial rhythm', 'multifocal ectopic atrial rhythm', 'multifocal ear', 'multifocal ear', 'dual atrial foci ', 'dual atrial foci ', 'multiple atrial foci', 'multiple atrial foci', 'multifocal atrial rhythm', 'multifocal atrial rhythm', 'multifocal atrialrhythm', 'multifocal atrialrhythm'], 'wandering': ['wandering atrial tachycardia', 'wandering atrial tachycardia', 'wandering ectopic atrial rhythm', 'wandering ectopic atrial rhythm', 'wandering ear', 'wandering ear', 'wandering atrial pacemaker', 'wandering atrial pacemaker'], 'unspecified': ['atrial rhythm', 'atrial rhythm', 'ectopic atrial rhythm', 'ectopic atrial rhythm', 'ear', 'ear', 'ectopic atrial bradycardia', 'ectopic atrial bradycardia', 'abnormal p vector', 'abnormal p vector', 'nonsinus atrial mechanism', 'nonsinus atrial mechanism', 'p wave axis suggests atrial rather than sinus mechanism', 'p wave axis suggests atrial rather than sinus mechanism', 'low atrial bradycardia', 'low atrial bradycardia']})) 

TMAPS['atrial_flutter'] = TensorMap('atrial_flutter', group='categorical', channel_map={'fixed_block': 0, 'variable_block': 1, 'unspecified': 2, }, tensor_from_file=make_partners_ecg_reads({'unspecified': ['fibrillation/flutter', 'fibrillation/flutter', 'aflutter', 'aflutter', 'atrial flutter', 'atrial flutter', 'probable flutter', 'probable flutter', 'atrial flutter unspecified block', 'atrial flutter unspecified block'], 'fixed_block': ['atrial flutter fixed block', 'atrial flutter fixed block'], 'variable_block': ['atrial flutter variable block', 'atrial flutter variable block']})) 

TMAPS['ectopic_atrial_tachycardia'] = TensorMap('ectopic_atrial_tachycardia', group='categorical', channel_map={'multifocal': 0, 'unifocal': 1, 'unspecified': 2, }, tensor_from_file=make_partners_ecg_reads({'unspecified': ['ectopic atrial tachycardia', 'ectopic atrial tachycardia', 'ectopic atrial tachycardia, unspecified', 'ectopic atrial tachycardia, unspecified', 'unspecified ectopic atrial tachycardia', 'unspecified ectopic atrial tachycardia'], 'unifocal': ['ectopic atrial tachycardia, unifocal', 'ectopic atrial tachycardia, unifocal', 'unifocal ectopic atrial tachycardia', 'unifocal ectopic atrial tachycardia'], 'multifocal': ['ectopic atrial tachycardia, multifocal', 'ectopic atrial tachycardia, multifocal', 'multifocal ectopic atrial tachycardia', 'multifocal ectopic atrial tachycardia']})) 

TMAPS['svt'] = TensorMap('svt', group='categorical', channel_map={'avrt': 0, 'avnrt': 1, 'unspecified': 2}, tensor_from_file=make_partners_ecg_reads({'avnrt': ['av nodal reentry tachycardia', 'av nodal reentry tachycardia', 'atrioventricular nodal reentry tachycardia', 'atrioventricular nodal reentry tachycardia', 'avnrt', 'avnrt'], 'avrt': ['atrioventricular reentrant tachycardia ', 'atrioventricular reentrant tachycardia ', 'av reentrant tachycardia ', 'av reentrant tachycardia ', 'avrt', 'avrt']})) 

