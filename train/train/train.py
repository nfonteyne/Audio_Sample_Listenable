import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

from preprocessing.preprocessing.utils import PreprocessingDataset
from sklearn.model_selection import train_test_split
from tensorflow import keras



def train(sound_path : str, marks_path : str,  metrics : str, test_size=0.2, separator=';'):

    def get_data():

        def get_sheet(file_path : str, sheet_name : str) -> pd.DataFrame():
            """
            Get excel sheet into a pandas dataframe and keeps only 'filename' and 'mediane' columns
            """
            sheet = pd.read_excel(file_path, sheet_name = sheet_name)
            sheet = sheet[[sheet_name, "Médiane", "Moyenne"]]
            sheet.columns = ["filename", "mediane", 'moyenne']
            return sheet
        
        sheet0 = get_sheet(file_path='fake', sheet_name='Sample 0000')
        sheet1 = get_sheet(file_path='fake', sheet_name='Sample 0001')
        sheet2 = get_sheet(file_path='fake', sheet_name='Sample 0002')
        sheet3 = get_sheet(file_path='fake', sheet_name='Sample 0003')
        sheet4 = get_sheet(file_path='fake', sheet_name='Sample 0004')
        sheet5 = get_sheet(file_path='fake', sheet_name='Sample 0005')
        sheet6 = get_sheet(file_path='fake', sheet_name='Sample 0006')
        sheet7 = get_sheet(file_path='fake', sheet_name='Sample 0007')
        sheet8 = get_sheet(file_path='fake', sheet_name='Sample 0008')
        sheet9 = get_sheet(file_path='fake', sheet_name='Sample 0009')

        full_df = pd.concat([sheet0, sheet1, sheet2, sheet3, sheet4, sheet5, sheet6, sheet7, sheet8, sheet9], ignore_index=True)
        
        full_df["bool_audible"] = [1 if i > 0 else 0 for i in full_df.mediane]

        full_df = full_df.replace({'mediane':{-2:0, -1.5:0, -1:1, -0.5:1, 0:2, 0.5:3, 1:3, 1.5:3, 2:4}})
        full_df['mediane'] = pd.to_numeric(full_df['mediane'], downcast='integer')
        
        full_df['filename'] = full_df['filename'].str.replace(r'-;$', '.wav')
        full_df['filename'] = full_df['filename'].str.replace(r'-$', '.wav')
        full_df['filename'] = full_df['filename'].map(lambda x: str(x)[1:])

        full_df.to_csv("Sample.csv", index=False)


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
