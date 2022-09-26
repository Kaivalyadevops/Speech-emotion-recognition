"""
This files creates the X and y features in joblib to be used by the predictive models.
"""

import os
import time
import joblib
import librosa
import numpy as np

TRAINING_FILES_PATH = './features/'
SAVE_DIR_PATH = './dataset_features/'


def features_maker(path, save_dir) -> str:
    """
    This function creates the dataset and saves both data and labels in
    two files, X.joblib and y.joblib in the joblib_features folder.
    """

    lst = []

    start_time = time.time()

    for subdir, dirs, files in os.walk(path):
        for file in files:
            try:
                # Load librosa array, obtain mfcss, store the file and the mcss information in a new array
                X, sample_rate = librosa.load(os.path.join(subdir, file),
                                                res_type='kaiser_fast')
                mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate,
                                                        n_mfcc=40).T, axis=0)
                # convert the labels (from 1 to 8) to a series from 0 to 7
                file = int(file[7:8]) - 1
                arr = mfccs, file
                lst.append(arr)
            # If the file is not valid, skip it
            except ValueError as err:
                print(err)
                continue

    print("--- Data loaded. Loading time: %s seconds ---" %
            (time.time() - start_time))

    # Creating X and y: zip makes a list of all the first elements, and a list of all the second elements.
    X, y = zip(*lst)

    X, y = np.asarray(X), np.asarray(y)

    print(X.shape, y.shape)

    X_name, y_name = 'X.joblib', 'y.joblib'

    joblib.dump(X, os.path.join(save_dir, X_name))
    joblib.dump(y, os.path.join(save_dir, y_name))

    return "Completed"


if __name__ == '__main__':
    print('Routine started')
    FEATURES = features_maker(
        path=TRAINING_FILES_PATH, save_dir=SAVE_DIR_PATH)
    print('Routine completed.')
