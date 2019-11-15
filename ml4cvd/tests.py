import unittest

from keras.losses import logcosh

from ml4cvd.TensorMap import TensorMap
from ml4cvd.arguments import parse_args
from ml4cvd.tensor_maps_by_script import TMAPS
from ml4cvd.DatabaseClient import BigQueryDatabaseClient, SqLiteDatabaseClient
from ml4cvd.recipes import test_multimodal_multitask, train_multimodal_multitask

ALL_TENSORS = '/mnt/ml4cvd/projects/tensors/sax-lax-ecg-rest-brain-1k/2019-11-06/'
ALL_TENSORS = '/mnt/disks/sax-lax-40k/2019-11-08/'
MODELS = '/mnt/ml4cvd/projects/models/'


def _run_tests():
    suites = []
    #suites.append(unittest.TestLoader().loadTestsFromTestCase(TestTensorMaps))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(TestTrainingModels))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(TestPretrainedModels))

    unittest.TextTestRunner(verbosity=3).run(unittest.TestSuite(suites))


class TestTensorMaps(unittest.TestCase):

    def test_tensor_map_equality(self):
        tensor_map_1a = TensorMap(name='tm', loss='logcosh', channel_map={'c1': 1, 'c2': 2}, metrics=[])
        tensor_map_1b = TensorMap(name='tm', loss='logcosh', channel_map={'c1': 1, 'c2': 2}, metrics=[])
        tensor_map_2a = TensorMap(name='tm', loss=logcosh, channel_map={'c1': 1, 'c2': 2}, metrics=[])
        tensor_map_2b = TensorMap(name='tm', loss=logcosh, channel_map={'c2': 2, 'c1': 1}, metrics=[])
        tensor_map_3 = TensorMap(name='tm', loss=logcosh, channel_map={'c1': 1, 'c2': 3}, metrics=[])
        tensor_map_4 = TensorMap(name='tm', loss=logcosh, channel_map={'c1': 1, 'c2': 3}, metrics=[all])
        tensor_map_5a = TensorMap(name='tm', loss=logcosh, channel_map={'c1': 1, 'c2': 3}, metrics=[all, any])
        tensor_map_5b = TensorMap(name='tm', loss=logcosh, channel_map={'c1': 1, 'c2': 3}, metrics=[any, all])
        tensor_map_6a = TensorMap(name='tm', loss=logcosh, channel_map={'c1': 1, 'c2': 3}, dependent_map=tensor_map_1a)
        tensor_map_6b = TensorMap(name='tm', loss=logcosh, channel_map={'c1': 1, 'c2': 3}, dependent_map=tensor_map_1b)

        self.assertEqual(tensor_map_1a, tensor_map_1b)
        self.assertEqual(tensor_map_2a, tensor_map_2b)
        self.assertEqual(tensor_map_1a, tensor_map_2a)
        self.assertNotEqual(tensor_map_2a, tensor_map_3)
        self.assertNotEqual(tensor_map_3, tensor_map_4)
        self.assertNotEqual(tensor_map_3, tensor_map_5a)
        self.assertNotEqual(tensor_map_4, tensor_map_5a)
        self.assertEqual(tensor_map_5a, tensor_map_5b)
        self.assertEqual(tensor_map_6a, tensor_map_6b)


class TestTrainingModels(unittest.TestCase):

    def test_train_categorical_mlp(self):
        delta = 1e-1
        args = parse_args()
        args.tensors = ALL_TENSORS
        args.input_tensors = ['categorical-phenotypes-134']
        args.output_tensors = ['coronary_artery_disease_soft', 'diabetes_type_2',
                               'hypertension', 'myocardial_infarction']
        args.epochs = 1
        args.batch_size = 32
        args.training_steps = 20
        args.validation_steps = 1
        args.test_steps = 32
        args.tensor_maps_in = [TMAPS[it] for it in args.input_tensors]
        args.tensor_maps_out = [TMAPS[ot] for ot in args.output_tensors]
        performances = train_multimodal_multitask(args)
        print('cat mlp expected = ', performances)
        expected = {'no_coronary_artery_disease_soft': 0.5280473195567535, 'coronary_artery_disease_soft': 0.5280473195567534,
                    'no_diabetes_type_2': 0.5175564681724847, 'diabetes_type_2': 0.5175564681724846, 'no_hypertension': 0.49742043019287246,
                    'hypertension': 0.49742043019287246, 'no_myocardial_infarction': 0.4442053930005737, 'myocardial_infarction': 0.44420539300057377}

        for k in performances:
            self.assertAlmostEqual(performances[k], expected[k], delta=delta)

    def test_train_mri_sax_zoom(self):
        delta = 7e-1
        args = parse_args()
        args.tensors = ALL_TENSORS
        args.input_tensors = ['sax_inlinevf_zoom_weighted']
        args.output_tensors = ['sax_inlinevf_zoom_mask_weighted', 'sex']
        args.epochs = 1
        args.batch_size = 4
        args.training_steps = 24
        args.validation_steps = 1
        args.test_steps = 36
        args.t = 48
        args.pool_z = 2
        args.u_connect = True
        args.learning_rate = 0.0001
        args.tensor_maps_in = [TMAPS[it] for it in args.input_tensors]
        args.tensor_maps_out = [TMAPS[ot] for ot in args.output_tensors]
        performances = train_multimodal_multitask(args)
        print('expected = ', performances)
        # expected = {'end_systole_volume_pearson': 0.010191148347492063,
        #             'end_diastole_volume_pearson': 0.07746212713601273,
        #             'ejection_fraction_pearson': 0.039482962469710864, 'no_allergic_rhinitis': 0.6583710407239819,
        #             'allergic_rhinitis': 0.655697243932538, 'no_asthma': 0.4921536796536796,
        #             'asthma': 0.49607683982683987, 'no_atrial_fibrillation_or_flutter': 0.3950320512820513,
        #             'atrial_fibrillation_or_flutter': 0.3950320512820512, 'no_back_pain': 0.5713760117733627,
        #             'back_pain': 0.5673289183222958, 'no_breast_cancer': 0.32195723684210525,
        #             'breast_cancer': 0.3223684210526316, 'no_coronary_artery_disease_soft': 0.37866666666666665,
        #             'coronary_artery_disease_soft': 0.37733333333333335, 'no_diabetes_type_2': 0.5410830999066294,
        #             'diabetes_type_2': 0.5401493930905695, 'no_hypertension': 0.5034782608695653,
        #             'hypertension': 0.5039613526570048, 'no_myocardial_infarction': 0.5632911392405063,
        #             'myocardial_infarction': 0.564873417721519}
        #
        # for k in expected:
        #     self.assertAlmostEqual(performances[k], expected[k], delta=delta)

    def test_train_mri_systole_diastole(self):
        delta = 6e-1
        args = parse_args()
        args.tensors = ALL_TENSORS
        args.input_tensors = ['mri_systole_diastole']
        args.output_tensors = ['corrected_extracted_lvedv', 'corrected_extracted_lvef', 'corrected_extracted_lvesv']
        args.epochs = 1
        args.batch_size = 12
        args.training_steps = 96
        args.validation_steps = 1
        args.test_steps = 36
        args.u_connect = True
        args.learning_rate = 0.0005
        args.tensor_maps_in = [TMAPS[it] for it in args.input_tensors]
        args.tensor_maps_out = [TMAPS[ot] for ot in args.output_tensors]
        performances = train_multimodal_multitask(args)
        print('expected = ', performances)
        # expected = {'background': 0.9997983811584907, 'ventricle': 0.3717467509311169, 'myocardium': 0.1753433358365212,
        #             'end_systole_volume_pearson': 0.1376, 'end_diastole_volume_pearson': 0.13085844389045515,
        #             'ejection_fraction_pearson': 0.09538719107146973, 'no_allergic_rhinitis': 0.46750762240688715,
        #             'allergic_rhinitis': 0.4675076224068871, 'no_asthma': 0.5619121188712681,
        #             'asthma': 0.5619121188712682, 'no_atrial_fibrillation_or_flutter': 0.6406823580220254,
        #             'atrial_fibrillation_or_flutter': 0.6406823580220254, 'no_back_pain': 0.5318302387267905,
        #             'back_pain': 0.5318302387267904, 'no_breast_cancer': 0.5274939903846154,
        #             'breast_cancer': 0.5274939903846154, 'no_coronary_artery_disease_soft': 0.589689578713969,
        #             'coronary_artery_disease_soft': 0.589689578713969, 'no_diabetes_type_2': 0.6338928856914469,
        #             'diabetes_type_2': 0.6338928856914469, 'no_hypertension': 0.5819672131147541,
        #             'hypertension': 0.5819672131147541, 'no_myocardial_infarction': 0.6285789335434726,
        #             'myocardial_infarction': 0.6285789335434726}
        #
        # for k in expected:
        #     self.assertAlmostEqual(performances[k], expected[k], delta=delta)


class TestPretrainedModels(unittest.TestCase):
    # def test_ecg_regress(self):
    #     delta = 1e-1
    #     args = parse_args()
    #     args.tensors = ALL_TENSORS
    #     args.model_file = MODELS + 'ecg_rest_regress/ecg_rest_regress.hd5'
    #     args.input_tensors = ['ecg_rest']
    #     args.output_tensors = ['p-axis', 'p-duration', 'p-offset', 'p-onset', 'pp-interval', 'pq-interval', 'q-offset', 'q-onset', 'qrs-complexes',
    #                            'qrs-duration', 'qt-interval', 'qtc-interval', 'r-axis', 'rr-interval', 't-offset', 't-axis']
    #     args.test_steps = 16
    #     args.batch_size = 24
    #     args.tensor_maps_in = [TMAPS[it] for it in args.input_tensors]
    #     args.tensor_maps_out = [TMAPS[ot] for ot in args.output_tensors]
    #     performances = test_multimodal_multitask(args)
    #     print('expected = ', performances)
    #     expected = {'PAxis_pearson': 0.6412350731914113, 'PDuration_pearson': 0.44687692331923495, 'POffset_pearson': 0.8895342855600766,
    #                 'POnset_pearson': 0.9497252876315257, 'PPInterval_pearson': 0.9832692070388677, 'PQInterval_pearson': 0.9301142630158935,
    #                 'QOffset_pearson': 0.7336190434160246, 'QOnset_pearson': 0.47727841194039183, 'QRSComplexes_pearson': 0.8786003993101409,
    #                 'QRSDuration_pearson': 0.7602037325063877, 'QTInterval_pearson': 0.947431443320984, 'QTCInterval_pearson': 0.9257252519356458,
    #                 'RAxis_pearson': 0.7788158778452872, 'RRInterval_pearson': 0.9852876188767442, 'TOffset_pearson': 0.9349277072650304,
    #                 'TAxis_pearson': 0.48564795968301755}
    #
    #     for k in expected:
    #         self.assertAlmostEqual(performances[k], expected[k], delta=delta)
    #
    # def test_ecg_rhythm(self):
    #     delta = 1e-1
    #     args = parse_args()
    #     args.tensors = ALL_TENSORS
    #     args.model_file = MODELS + 'ecg_rest_rhythm_hyperopted/ecg_rest_rhythm_hyperopted.hd5'
    #     args.input_tensors = ['ecg_rest']
    #     args.output_tensors = ['ecg_rhythm_poor']
    #     args.test_steps = 32
    #     args.batch_size = 32
    #     args.tensor_maps_in = [TMAPS[it] for it in args.input_tensors]
    #     args.tensor_maps_out = [TMAPS[ot] for ot in args.output_tensors]
    #     performances = test_multimodal_multitask(args)
    #     print('expected = ', performances)
    #     expected = {'Normal_sinus_rhythm': 0.9944891562668626, 'Sinus_bradycardia': 0.9986203969011992, 'Marked_sinus_bradycardia': 0.9998421717171717,
    #                 'Other_sinus_rhythm': 0.9789624183006536, 'Atrial_fibrillation': 0.9996513944223108, 'Other_rhythm': 0.9476284584980238}
    #
    #     for k in expected:
    #         self.assertAlmostEqual(performances[k], expected[k], delta=delta)
    #
    # def test_mri_systole_diastole_volumes(self):
    #     delta = 1e-1
    #     args = parse_args()
    #     args.tensors = ALL_TENSORS
    #     args.model_file = MODELS + 'mri_sd_unet_volumes/mri_sd_unet_volumes.hd5'
    #     args.input_tensors = ['mri_systole_diastole']
    #     args.output_tensors = ['mri_systole_diastole_segmented', 'corrected_extracted_lvedv', 'corrected_extracted_lvef', 'corrected_extracted_lvesv']
    #     args.optimizer = 'radam'
    #     args.test_steps = 32
    #     args.batch_size = 4
    #     args.tensor_maps_in = [TMAPS[it] for it in args.input_tensors]
    #     args.tensor_maps_out = [TMAPS[ot] for ot in args.output_tensors]
    #     performances = test_multimodal_multitask(args)
    #     print('expected = ', performances)
    #     expected = {'corrected_extracted_lvedv_pearson': 0.6844248954350666, 'corrected_extracted_lvef_pearson': 0.4995376682046898,
    #                 'corrected_extracted_lvesv_pearson': 0.6096212678064499}
    #
    #     for k in expected:
    #         self.assertAlmostEqual(performances[k], expected[k], delta=delta)

    def test_mri_systole_diastole_8_segment(self):
        delta = 1e-1
        args = parse_args()
        args.tensors = ALL_TENSORS
        args.model_file = MODELS + 'mri_sd8_unet/mri_sd8_unet.hd5'
        args.input_tensors = ['mri_systole_diastole_8_weighted']
        args.output_tensors = ['mri_systole_diastole_8_segmented_weighted']
        args.optimizer = 'radam'
        args.test_steps = 12
        args.batch_size = 4
        args.tensor_maps_in = [TMAPS[it] for it in args.input_tensors]
        args.tensor_maps_out = [TMAPS[ot] for ot in args.output_tensors]
        performances = test_multimodal_multitask(args)
        print('expected = ', performances)
        # expected = {'PAxis_pearson': 0.6115293530134417, 'PDuration_pearson': 0.5083110710202408, 'POffset_pearson': 0.8993388536229351,
        #             'POnset_pearson': 0.9456181625171349, 'PPInterval_pearson': 0.9876054363135571, 'PQInterval_pearson': 0.9012167913361175,
        #             'QOffset_pearson': 0.7678613436733094, 'QOnset_pearson': 0.5391954510894248, 'QRSComplexes_pearson': 0.9139094177062914,
        #             'QRSDuration_pearson': 0.7808130492073735, 'QTInterval_pearson': 0.9611575017458567, 'QTCInterval_pearson': 0.9602835173702873,
        #             'RAxis_pearson': 0.7068845948538833, 'RRInterval_pearson': 0.9873076763693096, 'TOffset_pearson': 0.938712542605686,
        #             'TAxis_pearson': 0.47777416060424527}
        #
        # for k in expected:
        #     self.assertAlmostEqual(performances[k], expected[k], delta=delta)


# Back to the top!
if '__main__' == __name__:
    _run_tests()
