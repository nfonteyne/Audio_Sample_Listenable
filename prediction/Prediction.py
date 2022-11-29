#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import librosa
from tqdm import tqdm
from tensorflow import keras

def main():
    model = keras.models.load_model('my_model')
    feature_list = []
    print('Preparing feature dataset and labels.')
    for file in tqdm(os.listdir('./Sound_File')):
        # Skip if it's not a wav file
        if not file.endswith('.wav'):
            continue
        # Load audio and stretch it to length 1s
        
        audio_path = os.path.join('./Sound_File/', file)
        
        audio, sr = librosa.load(path=audio_path, sr=None)
        audio = librosa.effects.time_stretch(y=audio, rate=len(audio)/sr)
        # Calculate features and get the label from the filename
        mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512)
        mels_db = librosa.power_to_db(S=mels, ref=1.0)
        if mels_db.shape == (128,87):
            feature_list.append(mels_db.reshape((128, 87, 1)))

        
    features = np.array(feature_list)   
    x=model.predict(features)

    for i in range(len(x)): #Iterate through the result of our model to determine wether the sound is listenable or not
        
        if(x[i][0]>0.5):
            print(i," : ", "son écoutable")
        else:
            print(i," : ", "son non écoutable") 

if __name__ == "__main__":
    main()