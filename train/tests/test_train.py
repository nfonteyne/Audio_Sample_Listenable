import unittest
from unittest.mock import MagicMock
import tempfile

import pandas as pd

from train.train import train
from preprocessing.preprocessing import utils

#python -m unittest discover --top-level-directory=. --start-directory=./train/tests/

"""def load_dataset_mock():

    dataset = pd.DataFrame({
                'filename': ['000000.wav','000057.wav','000094.wav','000110.wav','000135.wav','004026.wav','008508.wav','013862.wav','018424.wav','022966.wav','027433.wav','032508.wav','032551.wav','032634.wav','032653.wav','038517.wav','038584.wav','043725.wav'],
                'mediane': [3,2,4,3,4,3,4,3,3,0,2,0,4,3,3,0,3,4],
                'bool_audible': [1,0,1,1,1,1,1,1,1,0,0,0,1,1,1,0,1,1]
            })"""

class TestTrain(unittest.TestCase):

    def test_train(self):

        params = {'TEST_SIZE' : 0.2,
                'BATCH_SIZE' : 4,
                'EPOCHS' : 2}

            # run a training
        train.train(sound_path='preprocessing/test_data/audio_files', marks_path='preprocessing/test_data/marks/test_marks.xlsx', train_conf=params, model_path='train/tests/test_model')

