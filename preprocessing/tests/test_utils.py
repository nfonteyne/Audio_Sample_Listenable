import unittest
import pandas as pd
import numpy as np

from preprocessing.preprocessing import utils

"""
Run the following command in a bash terminal at the root of the project to run tests : 

    python3 -m unittest discover --top-level-directory=. --start-directory=./preprocessing/tests/
"""


class TestPreprocessingDataset(unittest.TestCase):


    def test_feature_shape_get_mel_spectrogram(self):
        """
        Test if get_mel_spectrogram() feature_list return expected shape
        """
        base = utils.PreprocessingDataset(sound_path='preprocessing/test_data/audio_files', marks_path='preprocessing/test_data/marks/test_marks.xlsx')
        base.get_mel_spectrogram()
        expected = (18,128,87,1)
        self.assertEqual(base.feature_list.shape,expected)


    def test_label_list_shape_get_mel_spectrogram(self):
        """
        Test if get_mel_spectrogram() label_list return expected shape
        """
        base = utils.PreprocessingDataset(sound_path='preprocessing/test_data/audio_files', marks_path='preprocessing/test_data/marks/test_marks.xlsx')
        base.get_mel_spectrogram()
        expected = (18,)
        print(base.label_list)
        self.assertEqual(base.label_list.shape,np.array(expected))


    def test_label_list_get_mel_spectrogram(self):
        """
        Test if get_mel_spectrogram() label list is correct
        """
        base = utils.PreprocessingDataset(sound_path='preprocessing/test_data/audio_files', marks_path='preprocessing/test_data/marks/test_marks.xlsx')
        base.get_mel_spectrogram()
        result = np.sort(base.label_list)
        expected = np.array([0,57,94,110,135,147,189,249,273,286,302,310,440,476])
        self.assertEqual(result.all(),expected.all())


    def test_matrix_shape_get_mel_spectrogram(self):
        """
        Test if get_mel_spectrogram() matrix return expected shape
        """
        base = utils.PreprocessingDataset(sound_path='preprocessing/test_data/audio_files', marks_path='preprocessing/test_data/marks/test_marks.xlsx')
        base.get_mel_spectrogram()
        expected = (18,128,87)
        self.assertEqual(base.matrix.shape,expected)
    


    def test_get_target(self):
        """
        Test if get_target return expected dataframe
        """
        base = utils.PreprocessingDataset(sound_path='preprocessing/test_data/audio_files', marks_path='preprocessing/test_data/marks/test_marks.xlsx')
        base.get_target()
        expected = pd.DataFrame({
            'filename': ['000000.wav','000057.wav','000094.wav','000110.wav','000135.wav','004026.wav','008508.wav','013862.wav','018424.wav','022966.wav','027433.wav','032508.wav','032551.wav','032634.wav','032653.wav','038517.wav','038584.wav','043725.wav'],
            'mediane': [3,2,4,3,4,3,4,3,3,0,2,0,4,3,3,0,3,4],
            'bool_audible': [1,0,1,1,1,1,1,1,1,0,0,0,1,1,1,0,1,1]
        })
        expected['mediane'] = pd.to_numeric(expected['mediane'], downcast='integer')
        pd.testing.assert_frame_equal(base.dataset,expected)

    
    def test_X_y_same_lenght(self):
        """
        Test if X and y are the same lenght
        """
        base = utils.PreprocessingDataset(sound_path='preprocessing/test_data/audio_files', marks_path='preprocessing/test_data/marks/test_marks.xlsx')
        X = base.matrix
        y = base.dataset['bool_audible']
        self.assertEqual(X.shape[0],y.shape[0])


    def test_y_shape(self):
        """
        Test if get_mel_spectrogram() targets return expected shape
        """
        base = utils.PreprocessingDataset(sound_path='preprocessing/test_data/audio_files', marks_path='preprocessing/test_data/marks/test_marks.xlsx')
        y = base.dataset['bool_audible']
        expected = (18,)
        self.assertEqual(y.shape,expected)
    