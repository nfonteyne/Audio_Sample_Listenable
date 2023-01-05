#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import librosa
from tqdm import tqdm
from tensorflow import keras
import streamlit as st


def save_uploadedfile(uploadedfile):
    with open(os.path.join("./Sound_File/", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File:{} to Data".format(uploadedfile.name))

def main():
    model = keras.models.load_model('my_model')
    feature_list = []
    Name_list=[]

    st.title('Bienvenue sur l\'application prediction !')
    st.text("Ici vous pourrez uploader des sons sous format .wav qui seront directement notés")
    st.text(" par notre algorithme.")

    Files=st.file_uploader("Uploadez vos ficher en .wav :",accept_multiple_files=True)
    
    if Files is not None:
        for file in Files:
            save_uploadedfile(file)

    else:
        st.text('You have not uploaded any files yet')            


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
            Name_list.append(file)

    if(len(feature_list)!=0):
        features = np.array(feature_list)   
        x=model.predict(features)
        for i in range(len(x)):
        
            if(x[i][0]>0.5):
                st.text(body=('le son ',Name_list[i],' est écoutable'))
                print(i," : ", "son écoutable")
            else:
                st.text(body=('le son ',Name_list[i],' n\'est pas écoutable'))
                print(i," : ", "son non écoutable") 

    else:
        st.write("Aucun fichier selectionné pour le moment")



    
    if st.button('Supprimer les fichier'):
        for file in tqdm(os.listdir('./Sound_File')):
            if file == None:
                continue
            audio_path = os.path.join('./Sound_File/', file)
            os.remove(audio_path)

    #Now that we've computed the prediction, we will create a website using streamlit to make predicting sounds easy

    



    

if __name__ == "__main__":
    main()