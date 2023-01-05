import unittest
import pandas as pd
from unittest.mock import MagicMock

from preprocessing.preprocessing import utils

"""
Run the following comand in a bash terminal at the root of the project to run tests : 

    python3 -m unittest discover --top-level-directory=. --start-directory=./preprocessing/tests/
"""


class TestPreprocessingDataset(unittest.TestCase):

    def test_get_mel_spectrogram():
        """
        Test if get_mel_spectrogram() return expected shape
        """
        base = utils.PreprocessingDataset()
        features, labels = base.get_mel_spectrogram()