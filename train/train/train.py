import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import os
import time
import json


from preprocessing.preprocessing.utils import PreprocessingDataset
from sklearn.model_selection import train_test_split
from tensorflow import keras

def train(sound_path : str, marks_path : str, train_conf : dict, model_path : str):
    """
    Train a Convolutional Neural Network model

    The model will be saved at ./prediction_streamlit/models

    If you run the model with Visual Studio code chances are that the program has dificulties to import 'preprocessing' module.
    You can bypass this error by running an unittest directly on the training data.
    Go in test_train.py for further informations.
    """

    # Get preprocessed dataset
    preprocessed_data = PreprocessingDataset(sound_path=sound_path,
                                            marks_path=marks_path)

    X = preprocessed_data.matrix
    y = preprocessed_data.dataset[['bool_audible']]

    # Split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_conf['TEST_SIZE'], random_state=123)

    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    # Instanciate the model
    model = keras.Sequential(layers=[
            keras.layers.InputLayer(input_shape=preprocessed_data.feature_list[0].shape),
            keras.layers.Conv2D(16, 3, padding='same', activation=keras.activations.relu),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(32, 3, padding='same', activation=keras.activations.relu),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation=keras.activations.relu),
            keras.layers.Dense(1, activation=keras.activations.sigmoid)
        ])
    
    # Compile model and display its summary in the terminal
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.binary_crossentropy, metrics=['accuracy'])
    print(model.summary())

    # Create callbacks
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train the model
    history = model.fit(x=X_train, y=y_train, validation_split=train_conf['TEST_SIZE'], epochs=train_conf['EPOCHS'], batch_size=train_conf['BATCH_SIZE'], callbacks=[early_stopping])


    # Create folder artefacts_path
    artefacts_path = os.path.join(model_path, time.strftime('%Y-%m-%d-%H-%M-%S'))
    os.mkdir(artefacts_path)

    # Save model in artefacts folder, name model.h5
    model.save(artefacts_path)

    # Save train_conf used in artefacts_path/params.json
    with open(artefacts_path + "/params.json", "w") as outfile:
        json.dump(train_conf, outfile)
    

    # Plot loss and accuracy
    fig, axs = plt.subplots(2)
    fig.set_size_inches(12, 8)
    fig.suptitle('Training History', fontsize=16)
    axs[0].plot(history.epoch, history.history['loss'], history.history['val_loss'])
    axs[0].set(title='Loss', xlabel='Epoch', ylabel='Loss')
    axs[0].legend(['loss', 'val_loss'])
    axs[1].plot(history.epoch, history.history['accuracy'], history.history['val_accuracy'])
    axs[1].set(title='Accuracy', xlabel='Epoch', ylabel='Accuracy')
    axs[1].legend(['accuracy', 'val_accuracy'])
    plt.show()

if __name__ == "__main__":

    # Instanciate hyperparameters
    params = {'TEST_SIZE' : 0.2,
            'BATCH_SIZE' : 25,
            'EPOCHS' : 20}

    # Train model
    train(sound_path='train/training_data/audio', marks_path='train/training_data/marks/Notes_des_sons.xlsx', train_conf=params, model_path='prediction_streamlit/models')