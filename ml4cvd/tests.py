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
    #suites.append(unittest.TestLoader().loadTestsFromTestCase(TestTrainingModels))
    # TODO Add them to 'suites' when the pretrained model tests are updated to work with the tensors at ALL_TENSORS
    suites.append(unittest.TestLoader().loadTestsFromTestCase(TestPretrainedModels))
    # suites.append(unittest.TestLoader().loadTestsFromTestCase(TestDatabaseClient))

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
        args.input_tensors = ['categorical-phenotypes-78']
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
        print('expected = ', performances)
        expected = {'no_coronary_artery_disease_soft': 0.528143258213825, 'coronary_artery_disease_soft': 0.528143258213825,
                    'no_diabetes_type_2': 0.6547365677800461, 'diabetes_type_2': 0.654736567780046, 'no_hypertension': 0.4729961761211761,
                    'hypertension': 0.4729961761211761, 'no_myocardial_infarction': 0.5480460307260938, 'myocardial_infarction': 0.5480460307260938}

        for k in performances:
            self.assertAlmostEqual(performances[k], expected[k], delta=delta)

    def test_train_mlp_cat36_pi(self):
        delta = 8e-1
        args = parse_args()
        args.tensors = ALL_TENSORS
        args.input_tensors = ['categorical-phenotypes-36']
        args.output_tensors = ['end_systole_volume', 'end_diastole_volume', 'ejection_fractionp',
                               'allergic_rhinitis_prevalent_incident', 'asthma_prevalent_incident',
                               'atrial_fibrillation_or_flutter_prevalent_incident', 'back_pain_prevalent_incident',
                               'breast_cancer_prevalent_incident', 'coronary_artery_disease_soft_prevalent_incident',
                               'diabetes_type_2_prevalent_incident',
                               'hypertension_prevalent_incident', 'myocardial_infarction_prevalent_incident']
        args.epochs = 1
        args.batch_size = 32
        args.training_steps = 20
        args.validation_steps = 1
        args.test_steps = 32
        args.tensor_maps_in = [TMAPS[it] for it in args.input_tensors]
        args.tensor_maps_out = [TMAPS[ot] for ot in args.output_tensors]
        performances = train_multimodal_multitask(args)
        print('expected = ', performances)
        expected = {'end_systole_volume_pearson': -0.18898451656690726,
                    'end_diastole_volume_pearson': 0.21810526919810447,
                    'ejection_fraction_pearson': 0.10870165257050499, 'no_allergic_rhinitis': 0.6144818514407859,
                    'prevalent_allergic_rhinitis': 0.7167318982387475, 'incident_allergic_rhinitis': 0.6180602006688962,
                    'no_asthma': 0.63458251953125, 'prevalent_asthma': 0.39872051795899494,
                    'incident_asthma': 0.6231338522393773, 'no_atrial_fibrillation_or_flutter': 0.5410833333333334,
                    'prevalent_atrial_fibrillation_or_flutter': 0.4638504611330698,
                    'incident_atrial_fibrillation_or_flutter': 0.633399209486166, 'no_back_pain': 0.5106227106227106,
                    'prevalent_back_pain': 0.5311572700296736, 'incident_back_pain': 0.6132478632478633,
                    'no_breast_cancer': 0.4686281065418478, 'prevalent_breast_cancer': 0.41631089217296113,
                    'incident_breast_cancer': 0.5118179810028718, 'no_coronary_artery_disease_soft': 0.527491408934708,
                    'prevalent_coronary_artery_disease_soft': 0.6513790600616949,
                    'incident_coronary_artery_disease_soft': 0.43841355846774194,
                    'no_diabetes_type_2': 0.5623635418471669, 'prevalent_diabetes_type_2': 0.48804846103470856,
                    'incident_diabetes_type_2': 0.5779872301611432, 'no_hypertension': 0.4664909906701802,
                    'prevalent_hypertension': 0.5800677998245739, 'incident_hypertension': 0.47476802434951704,
                    'no_myocardial_infarction': 0.6280876494023904,
                    'prevalent_myocardial_infarction': 0.7055550569864488,
                    'incident_myocardial_infarction': 0.5212917350848385}

        for k in performances:
            self.assertAlmostEqual(performances[k], expected[k], delta=delta)

    def test_train_mri_sax_zoom(self):
        delta = 7e-1
        args = parse_args()
        args.tensors = ALL_TENSORS
        args.input_tensors = ['sax_inlinevf_zoom_weighted']
        args.output_tensors = ['sax_inlinevf_zoom_mask_weighted', 'end_systole_volume', 'end_diastole_volume',
                               'ejection_fractionp', 'allergic_rhinitis', 'asthma', 'atrial_fibrillation_or_flutter',
                               'back_pain', 'breast_cancer', 'coronary_artery_disease_soft', 'diabetes_type_2',
                               'hypertension', 'myocardial_infarction']
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
        expected = {'end_systole_volume_pearson': 0.010191148347492063,
                    'end_diastole_volume_pearson': 0.07746212713601273,
                    'ejection_fraction_pearson': 0.039482962469710864, 'no_allergic_rhinitis': 0.6583710407239819,
                    'allergic_rhinitis': 0.655697243932538, 'no_asthma': 0.4921536796536796,
                    'asthma': 0.49607683982683987, 'no_atrial_fibrillation_or_flutter': 0.3950320512820513,
                    'atrial_fibrillation_or_flutter': 0.3950320512820512, 'no_back_pain': 0.5713760117733627,
                    'back_pain': 0.5673289183222958, 'no_breast_cancer': 0.32195723684210525,
                    'breast_cancer': 0.3223684210526316, 'no_coronary_artery_disease_soft': 0.37866666666666665,
                    'coronary_artery_disease_soft': 0.37733333333333335, 'no_diabetes_type_2': 0.5410830999066294,
                    'diabetes_type_2': 0.5401493930905695, 'no_hypertension': 0.5034782608695653,
                    'hypertension': 0.5039613526570048, 'no_myocardial_infarction': 0.5632911392405063,
                    'myocardial_infarction': 0.564873417721519}

        for k in expected:
            self.assertAlmostEqual(performances[k], expected[k], delta=delta)

    def test_train_mri_systole_diastole(self):
        delta = 6e-1
        args = parse_args()
        args.tensors = ALL_TENSORS
        args.input_tensors = ['mri_systole_diastole_weighted']
        args.output_tensors = ['mri_systole_diastole_segmented_weighted', 'end_systole_volume', 'end_diastole_volume',
                               'ejection_fractionp', 'allergic_rhinitis', 'asthma', 'atrial_fibrillation_or_flutter',
                               'back_pain', 'breast_cancer', 'coronary_artery_disease_soft', 'diabetes_type_2',
                               'hypertension', 'myocardial_infarction']
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
        expected = {'background': 0.9997983811584907, 'ventricle': 0.3717467509311169, 'myocardium': 0.1753433358365212,
                    'end_systole_volume_pearson': 0.1376, 'end_diastole_volume_pearson': 0.13085844389045515,
                    'ejection_fraction_pearson': 0.09538719107146973, 'no_allergic_rhinitis': 0.46750762240688715,
                    'allergic_rhinitis': 0.4675076224068871, 'no_asthma': 0.5619121188712681,
                    'asthma': 0.5619121188712682, 'no_atrial_fibrillation_or_flutter': 0.6406823580220254,
                    'atrial_fibrillation_or_flutter': 0.6406823580220254, 'no_back_pain': 0.5318302387267905,
                    'back_pain': 0.5318302387267904, 'no_breast_cancer': 0.5274939903846154,
                    'breast_cancer': 0.5274939903846154, 'no_coronary_artery_disease_soft': 0.589689578713969,
                    'coronary_artery_disease_soft': 0.589689578713969, 'no_diabetes_type_2': 0.6338928856914469,
                    'diabetes_type_2': 0.6338928856914469, 'no_hypertension': 0.5819672131147541,
                    'hypertension': 0.5819672131147541, 'no_myocardial_infarction': 0.6285789335434726,
                    'myocardial_infarction': 0.6285789335434726}

        for k in expected:
            self.assertAlmostEqual(performances[k], expected[k], delta=delta)

    def test_train_mri_systole_diastole_pi(self):
        delta = 6e-1
        args = parse_args()
        args.tensors = ALL_TENSORS
        args.input_tensors = ['mri_systole_diastole_weighted']
        args.output_tensors = ['mri_systole_diastole_segmented_weighted', 'end_systole_volume', 'end_diastole_volume',
                               'ejection_fractionp', 'allergic_rhinitis_prevalent_incident',
                               'asthma_prevalent_incident', 'atrial_fibrillation_or_flutter_prevalent_incident',
                               'back_pain_prevalent_incident', 'breast_cancer_prevalent_incident',
                               'coronary_artery_disease_soft_prevalent_incident',
                               'diabetes_type_2_prevalent_incident', 'hypertension_prevalent_incident',
                               'myocardial_infarction_prevalent_incident']
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
        expected = {'background': 0.9994056020189077, 'ventricle': 0.3217267098772595, 'myocardium': 0.13751608768060483,
                    'end_systole_volume_pearson': -0.11803115474509973, 'end_diastole_volume_pearson': 0.07460391201856462,
                    'ejection_fraction_pearson': 0.014315858148787145, 'no_allergic_rhinitis': 0.4270341364261374,
                    'prevalent_allergic_rhinitis': 0.5918604651162791, 'incident_allergic_rhinitis': 0.437839186576009,
                    'no_asthma': 0.4421774889807788, 'prevalent_asthma': 0.4578408195429472, 'incident_asthma': 0.4612041884816754,
                    'no_atrial_fibrillation_or_flutter': 0.33988339451522354, 'prevalent_atrial_fibrillation_or_flutter': 0.3615023474178404,
                    'incident_atrial_fibrillation_or_flutter': 0.2651053864168618, 'no_back_pain': 0.5578817733990148,
                    'prevalent_back_pain': 0.307563025210084, 'incident_back_pain': 0.553459920988913, 'no_breast_cancer': 0.49939903846153844,
                    'prevalent_breast_cancer': 0.6088992974238876, 'incident_breast_cancer': 0.45130641330166266,
                    'no_coronary_artery_disease_soft': 0.4605321507760532, 'prevalent_coronary_artery_disease_soft': 0.3188862621486735,
                    'incident_coronary_artery_disease_soft': 0.49899026987332473, 'no_diabetes_type_2': 0.47961630695443647,
                    'prevalent_diabetes_type_2': 0.5485625485625485, 'incident_diabetes_type_2': 0.4376984126984127,
                    'no_hypertension': 0.6255949233209941, 'prevalent_hypertension': 0.4478044259066156, 'incident_hypertension': 0.6184678890849811,
                    'no_myocardial_infarction': 0.5187811925400578, 'prevalent_myocardial_infarction': 0.5925058548009368,
                    'incident_myocardial_infarction': 0.4287383177570093}

        for k in expected:
            self.assertAlmostEqual(performances[k], expected[k], delta=delta)


class TestPretrainedModels(unittest.TestCase):
    def test_ecg_regress(self):
        delta = 1e-1
        args = parse_args()
        args.tensors = ALL_TENSORS
        args.model_file = MODELS + 'ecg_rest_regress/ecg_rest_regress.hd5'
        args.input_tensors = ['ecg_rest']
        args.output_tensors = ['p-axis', 'p-duration', 'p-offset', 'p-onset', 'pp-interval', 'pq-interval', 'q-offset', 'q-onset', 'qrs-complexes',
                               'qrs-duration', 'qt-interval', 'qtc-interval', 'r-axis', 'rr-interval', 't-offset', 't-axis']
        args.test_steps = 16
        args.batch_size = 24
        args.tensor_maps_in = [TMAPS[it] for it in args.input_tensors]
        args.tensor_maps_out = [TMAPS[ot] for ot in args.output_tensors]
        performances = test_multimodal_multitask(args)
        print('expected = ', performances)
        expected = {'PAxis_pearson': 0.6096757853247863, 'PDuration_pearson': 0.5033754637888465, 'POffset_pearson': 0.8991941381015535,
                    'POnset_pearson': 0.943068274973917, 'PPInterval_pearson': 0.9741982064846891, 'PQInterval_pearson': 0.9183941991560995,
                    'QOffset_pearson': 0.6914958367104611, 'QOnset_pearson': 0.4973036541178359, 'QRSComplexes_pearson': 0.8454838977323635,
                    'QRSDuration_pearson': 0.6909425663163459, 'QTInterval_pearson': 0.9256624839421144, 'QTCInterval_pearson': 0.9156416484270498,
                    'RAxis_pearson': 0.7784796569323758, 'RRInterval_pearson': 0.9783352088344341, 'TOffset_pearson': 0.9245605316261704,
                    'TAxis_pearson': 0.47724645273243477}

        for k in expected:
            self.assertAlmostEqual(performances[k], expected[k], delta=delta)

    def test_ecg_rhythm(self):
        delta = 1e-1
        args = parse_args()
        args.tensors = ALL_TENSORS
        args.model_file = MODELS + 'ecg_rest_rhythm_hyperopted/ecg_rest_rhythm_hyperopted.hd5'
        args.input_tensors = ['ecg_rest']
        args.output_tensors = ['ecg_rhythm_poor']
        args.test_steps = 32
        args.batch_size = 32
        args.tensor_maps_in = [TMAPS[it] for it in args.input_tensors]
        args.tensor_maps_out = [TMAPS[ot] for ot in args.output_tensors]
        performances = test_multimodal_multitask(args)
        print('expected = ', performances)
        expected = {'Normal_sinus_rhythm': 0.995458984375, 'Sinus_bradycardia': 0.9980378995198017, 'Marked_sinus_bradycardia': 1.0,
                    'Other_sinus_rhythm': 0.9764925373134328, 'Atrial_fibrillation': 1.0, 'Other_rhythm': 0.9637426900584795}

        for k in expected:
            self.assertAlmostEqual(performances[k], expected[k], delta=delta)

    def test_mri_systole_diastole_volumes(self):
        delta = 1e-1
        args = parse_args()
        args.tensors = ALL_TENSORS
        args.model_file = MODELS + 'mri_sd_unet_volumes/mri_sd_unet_volumes.hd5'
        args.input_tensors = ['mri_systole_diastole']
        args.output_tensors = ['mri_systole_diastole_segmented', 'corrected_extracted_lvedv', 'corrected_extracted_lvef', 'corrected_extracted_lvesv']
        args.optimizer = 'radam'
        args.test_steps = 32
        args.batch_size = 4
        args.tensor_maps_in = [TMAPS[it] for it in args.input_tensors]
        args.tensor_maps_out = [TMAPS[ot] for ot in args.output_tensors]
        performances = test_multimodal_multitask(args)
        print('expected = ', performances)
        expected = {'corrected_extracted_lvedv_pearson': 0.6500756491729536, 'corrected_extracted_lvef_pearson': 0.4773548108871419,
                    'corrected_extracted_lvesv_pearson': 0.556143488570414}

        for k in expected:
            self.assertAlmostEqual(performances[k], expected[k], delta=delta)


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


class TestDatabaseClient(unittest.TestCase):
    def test_database_client(self):
        args = parse_args()

        bigquery_client = BigQueryDatabaseClient(credentials_file=args.bigquery_credentials_file)
        sqlite_client = SqLiteDatabaseClient(db_file=args.db)

        query = (
            'SELECT field, fieldid FROM {} '
            'WHERE fieldid BETWEEN 3120 AND 3190 '
            'ORDER BY field ASC '
            'LIMIT 7'
        )

        table = 'dictionary'
        dataset = args.bigquery_dataset

        bigquery_rows = bigquery_client.execute(query.format(f"`{dataset}.{table}`"))
        sqlite_rows = sqlite_client.execute(query.format(table))

        for bq_row, sql_row in zip(bigquery_rows, sqlite_rows):
            self.assertEqual((bq_row[0], bq_row[1]), sql_row)


# Back to the top!
if '__main__' == __name__:
    _run_tests()
