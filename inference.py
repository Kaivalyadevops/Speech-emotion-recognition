"""
This file can be used to try a live prediction. 
"""

import tensorflow
import numpy as np
import librosa


class liveSER:
    def __init__(self, path, file):
        self.path = path
        self.file = file

    def load_model(self):
        self.loaded_model = tensorflow.keras.models.load_model(self.path)
        return self.loaded_model

    def predictions(self):
        data, sampling_rate = librosa.load(self.file)
        mfccs = np.mean(librosa.feature.mfcc(
            y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        x = np.expand_dims(mfccs, axis=1)
        x = np.expand_dims(x, axis=0)
        predictions = self.loaded_model.predict_classes(x)
        print("Prediction is", " ", self.class_to_emotion(predictions))

    def class_to_emotion(self, pred):
        label_conversion = {'0': 'neutral',
                            '1': 'calm',
                            '2': 'happy',
                            '3': 'sad',
                            '4': 'angry',
                            '5': 'fearful',
                            '6': 'disgust',
                            '7': 'surprised'}

        for key, value in label_conversion.items():
            if int(key) == pred:
                label = value
        return label


pred = liveSER(path='Deep Learning/SER_model.h5',
               file='examples/10-16-07-29-82-30-63.wav')

pred.load_model()
pred.predictions()
