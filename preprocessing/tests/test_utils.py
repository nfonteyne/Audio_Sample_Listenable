import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from preprocessing.preprocessing import utils

"""
Run the following comand in a bash terminal at the root of the project to run tests : 

    python3 -m unittest discover --top-level-directory=. --start-directory=./preprocessing/tests/
"""


class TestPreprocessingDataset(unittest.TestCase):

    def test_feature_shape_get_mel_spectrogram(self):
        """
        Test if get_mel_spectrogram() feature_list return expected shape
        """
        base = utils.PreprocessingDataset(sound_path='preprocessing/test_data/audio_files', marks_path='preprocessing/test_data/marks/test_marks.csv', separator=';')
        base.get_mel_spectrogram()
        expected = (14,128,87,1)
        self.assertEqual(base.feature_list.shape,expected)

    def test_label_list_shape_get_mel_spectrogram(self):
        """
        Test if get_mel_spectrogram() label_list return expected shape
        """
        base = utils.PreprocessingDataset(sound_path='preprocessing/test_data/audio_files', marks_path='preprocessing/test_data/marks/test_marks.csv', separator=';')
        base.get_mel_spectrogram()
        expected = (14,)
        print(base.label_list)
        self.assertEqual(base.label_list.shape,np.array(expected))

    def test_label_list_get_mel_spectrogram(self):
        """
        Test if get_mel_spectrogram() return expected shape
        """
        base = utils.PreprocessingDataset(sound_path='preprocessing/test_data/audio_files', marks_path='preprocessing/test_data/marks/test_marks.csv', separator=';')
        base.get_mel_spectrogram()
        result = np.sort(base.label_list)
        expected = np.array([0,57,94,110,135,147,189,249,273,286,302,310,440,476])
        self.assertEqual(result.all(),expected.all())

    