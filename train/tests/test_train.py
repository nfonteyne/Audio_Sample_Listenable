import unittest
from unittest.mock import MagicMock

import pandas as pd

from train.train import train

#python -m unittest discover --top-level-directory=. --start-directory=./train/tests/

class TestTrain(unittest.TestCase):

    def test_train(self):

        params = {'TEST_SIZE' : 0.2,
                'BATCH_SIZE' : 4,
                'EPOCHS' : 2}

            # run a training
        train.train(sound_path='preprocessing/test_data/audio_files', marks_path='preprocessing/test_data/marks/test_marks.xlsx', train_conf=params, model_path='train/tests/test_model')


        """

        params = {'TEST_SIZE' : 0.2,
                'BATCH_SIZE' : 25,
                'EPOCHS' : 20}

        train.train(sound_path='train/training_data/audio', marks_path='train/training_data/marks/Notes_des_sons.xlsx', train_conf=params, model_path='prediction_streamlit/models')
        """
