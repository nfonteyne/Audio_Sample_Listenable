import librosa
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

class PreprocessingDataset():

    def __init__(self):
        
        self.feature_list = []
        self.label_list = []
        self.dictMel = {}
        self.matrix = np.empty()
        self.listMel = []
        self.liste_son = []
        self.dataset = pd.DataFrame()

    def get_mel_spectrogram(self):
        """
        Return features and labels of all .wav sound files in 'Sound_File' folder
        """
        # Iterate over all files in given source path
        print('Preparing feature dataset and labels.')
        
        for file in tqdm(os.listdir('preprocessing/test_data/audio_files')):
            # Skip if it's not a wav file
            if not file.endswith('.wav'):
                continue
            
            # Load audio and stretch it to length 1s
            audio_path = os.path.join('preprocessing/test_data/audio_files', file)
            audio, sr = librosa.load(path=audio_path, sr=None)
            audio = librosa.effects.time_stretch(y=audio, rate=len(audio)/sr)
            
            # Calculate features and get the label from the filename
            mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512)
            mels_db = librosa.power_to_db(S=mels, ref=1.0)
            
            if mels_db.shape == (128,87):
                self.feature_list.append(mels_db.reshape((128, 87, 1)))
                filename = int(str(file)[:-4])
                self.label_list.append(filename)
                self.dictMel[filename] = mels_db
            
            self.feature_list = np.array(self.feature_list)
            self.label_list = np.array(self.label_list)


    def order_by_name(self):
        dataMel = pd.Series(self.dictMel)
        dataMel = dataMel.sort_index()
        self.listMel = dataMel.to_list()
        self.liste_son = dataMel.index.to_list()


    def list_to_matrix(self):
        self.matrix = np.zeros(shape=(len(self.listMel),128, 87))
        for i in range(len(self.listMel)):
            if self.listMel[i].shape == (128, 87):
                self.matrix[i]=self.listMel[i]

    def get_target(self):
        ds = pd.read_csv("preprocessing/test_data/labels/test_labels.csv",sep=";")
        ds['nomSon'] = ds['nomSon'].str.replace(r"[-;]", '',regex =True).astype(int)
        ds = ds[ds['nomSon'].isin(self.liste_son)]
        ds['Moyenne'] = ds['Moyenne'].str.replace(',','.')
        ds['Moyenne'] = ds['Moyenne'].astype(float)
        ds.Moyenne = [0 if i<-1.2 else 1 if i<-0.4 else 2 if i<0.4 else 3 if i<1.2 else 4 for i in ds.Moyenne]
        ds['MedianeBin'] = [0 if i<1 else 1 for i in ds.Mediane]
        ds.Mediane = [0 if i==-2 else 1 if i==-1 else 2 if i==0 else 3 if i==1 else 4 for i in ds.Mediane]
        self.dataset = ds     

    def run(self):
        self.get_mel_spectrogram()
        self.order_by_name()
        self.list_to_matrix()
        self.get_target()