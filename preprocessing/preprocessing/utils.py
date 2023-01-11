import librosa
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

class PreprocessingDataset:

    def __init__(self, sound_path : str, marks_path : str):

        self.feature_list = []
        self.label_list = []
        self.matrix = None
        self.listMel = []
        self.liste_son = []
        self.dataset = pd.DataFrame()

        self.audio_path = sound_path
        self.marks_path = marks_path

        self.get_mel_spectrogram()
        self.get_target()


    def get_mel_spectrogram(self):
        """
        Return features and labels of all .wav sound files in 'Sound_File' folder
        """
        # Iterate over all files in given source path
        print('Preparing feature dataset and labels.')

        feature_list = []
        label_list = []
        dictMel = {}
        
        for file in tqdm(os.listdir(self.audio_path)):
            # Skip if it's not a wav file
            if not file.endswith('.wav'):
                continue
            
            # Load audio and stretch it to length 1s
            audio_path = os.path.join(self.audio_path, file)
            audio, sr = librosa.load(path=audio_path, sr=None)
            audio = librosa.effects.time_stretch(y=audio, rate=len(audio)/sr)
            
            # Calculate features and get the label from the filename
            mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512)
            mels_db = librosa.power_to_db(S=mels, ref=1.0)
            
            if mels_db.shape == (128,87):
                feature_list.append(mels_db.reshape((128, 87, 1)))
                filename = int(str(file)[:-4])
                label_list.append(filename)
                dictMel[filename] = mels_db
            
            self.feature_list = np.array(feature_list)
            self.label_list = np.array(label_list)


        dataMel = pd.Series(dictMel)
        dataMel = dataMel.sort_index()
        self.listMel = dataMel.to_list()
        self.liste_son = dataMel.index.to_list()


        self.matrix = np.zeros(shape=(len(self.listMel),128, 87))

        for i in range(len(self.listMel)):
            if self.listMel[i].shape == (128, 87):
                self.matrix[i] = self.listMel[i]


    def get_target(self):
        """
        Get model's targets from excel file into a pandas dataframe
        """

        def get_sheet(sheet_name : str, file_path = self.marks_path) -> pd.DataFrame:
            """
            Get excel sheet into a pandas dataframe and keeps only 'filename' and 'mediane' columns
            """
            sheet = pd.read_excel(file_path, sheet_name = sheet_name)
            sheet = sheet[[sheet_name, "Médiane"]]
            sheet.columns = ["filename", "mediane"]
            return sheet
        
        sheet0 = get_sheet(sheet_name='Sample 0000')
        sheet1 = get_sheet(sheet_name='Sample 0001')
        sheet2 = get_sheet(sheet_name='Sample 0002')
        sheet3 = get_sheet(sheet_name='Sample 0003')
        sheet4 = get_sheet(sheet_name='Sample 0004')
        sheet5 = get_sheet(sheet_name='Sample 0005')
        sheet6 = get_sheet(sheet_name='Sample 0006')
        sheet7 = get_sheet(sheet_name='Sample 0007')
        sheet8 = get_sheet(sheet_name='Sample 0008')
        sheet9 = get_sheet(sheet_name='Sample 0009')

        full_df = pd.concat([sheet0, sheet1, sheet2, sheet3, sheet4, sheet5, sheet6, sheet7, sheet8, sheet9], ignore_index=True)
        

        full_df["bool_audible"] = [1 if i > 0 else 0 for i in full_df.mediane]

        full_df = full_df.replace({'mediane':{-2:0, -1.5:0, -1:1, -0.5:1, 0:2, 0.5:3, 1:3, 1.5:3, 2:4}})
        full_df['mediane'] = pd.to_numeric(full_df['mediane'], downcast='integer')
            
        full_df['filename'] = full_df['filename'].str.replace(r'-;$', '.wav')
        full_df['filename'] = full_df['filename'].str.replace(r'-$', '.wav')
        full_df['filename'] = full_df['filename'].map(lambda x: str(x)[1:])

        self.dataset = full_df