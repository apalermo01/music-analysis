import numpy as np
import librosa
import tarfile
import soundfile as sf
import io
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


class MfccPipeline():

    def __init__(self):
        self.PATH = "C:/Users/alexc/Downloads/nsynth-train.jsonwav.tar.gz"

        self.targets_list = [
            "bass", "brass", "flute", "guitar", "keyboard",
            "mallet", "organ", "reed", "string", "synth", "vocal"
        ]

    # recover wav files from tarball
    def get_files(self, PATH=None, num_files = 100):
        """Sequentially pull wav files from .tar.gz file
        :param PATH: path to compressed dataset file
        :param num_files: number of files to read
        :returns: 4 generators for data (wav file converted to list), sr (sample rate)
                target name (e.g. 'guitar', 'mallet', ect.), and target index
        """

        print("getting files")
        if PATH is None:
            PATH = self.PATH

        # open the tar file
        with tarfile.open(PATH, 'r:gz') as tar:

            # Initialize counter to count number of files pulled.
            index = 0
            while index < num_files:
                fname = tar.next()

                # Break if there are no more files.
                if fname is None:
                    break

                # Check that we're dealing with the proper format
                if fname.name.endswith(".wav"):

                    # Extract file
                    wav_file = tar.extractfile(fname).read()

                    # Convert bytes to a readable format
                    data, sr = sf.read(io.BytesIO(wav_file))

                    # Get target from filename
                    target = fname.name.split('/')[2].split('_')[0]

                    # yeild the 4 generators
                    yield data, sr, target, self.targets_list.index(target)
                    index += 1

    # collect raw data into an array
    def get_dataset(self, num_files=10):
        """Docstring pending
        """
        data = []
        data_generator = self.get_files(self.PATH, num_files)
        for i in range(num_files):
            data.append(next(data_generator))
        return data

    # calculate mfccs
    def get_mfccs(self, data_tuple):
        """Take a tuple of data (data, sr, target, target_index) and return the associated mfcc"""
        data = np.array(data_tuple[0])
        sr = data_tuple[1]
        mfcc = librosa.feature.mfcc(y=data, sr=sr)
        return mfcc

    # prepare the data for input to model
    def prepare_data(self, dataset):
        """Get the mfccs for each record in the dataset and the associated target values
        """

        X = np.array([self.get_mfccs(i) for i in dataset])
        t = to_categorical(np.array([i[3] for i in dataset]))

        return X, t

    def mfcc_pipeline(self, num_samples, validation_split=0.2):
        """Write this doc later"""
        # get the data from tar file
        data = self.get_dataset(num_samples)
        X, t = self.prepare_data(data)

        # split into train and validation set and return the data
        return train_test_split(X, t, test_size=validation_split)