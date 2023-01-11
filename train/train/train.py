import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

from preprocessing.preprocessing.utils import PreprocessingDataset
from sklearn.model_selection import train_test_split
from tensorflow import keras



def train(sound_path : str, marks_path : str,  metrics : str, test_size=0.2, separator=';'):

    preprocessed_data = PreprocessingDataset(sound_path=sound_path,
                                            marks_path=marks_path,
                                            separator=separator)

    preprocessed_data.run()

    X = preprocessed_data.matrix
    y = preprocessed_data.dataset[[metrics]]
    test_size = test_size

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=123)

    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

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
    
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.binary_crossentropy, metrics=['accuracy'])
    print(model.summary())

    TEST_SIZE = 0.2
    BATCH_SIZE = 25
    EPOCHS = 20

    # Train the model
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(x=X_train, y=y_train, validation_split=TEST_SIZE, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping])


    
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
