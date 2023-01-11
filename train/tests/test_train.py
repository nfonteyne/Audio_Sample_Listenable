import unittest

from train.train import train

"""
Run the following command in a bash terminal at the root of the project to run tests : 

    python3 -m unittest discover --top-level-directory=. --start-directory=./train/tests/
"""


class TestTrain(unittest.TestCase):

    def test_train(self):
        """
        Run a tranning with a small dataset
        """

        params = {'TEST_SIZE' : 0.2,
                'BATCH_SIZE' : 4,
                'EPOCHS' : 2}

        train.train(sound_path='preprocessing/test_data/audio_files', marks_path='preprocessing/test_data/marks/test_marks.xlsx', train_conf=params, model_path='train/tests/test_model')


        """
        To bypass 'ModuleNotFoundError' on preprocessing module put the params and train.train above in comment and run the lines below as a unittest.

        params = {'TEST_SIZE' : 0.2,
                'BATCH_SIZE' : 25,
                'EPOCHS' : 20}

        train.train(sound_path='train/training_data/audio', marks_path='train/training_data/marks/Notes_des_sons.xlsx', train_conf=params, model_path='prediction_streamlit/models')
        """
