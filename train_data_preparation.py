import numpy as np
import librosa
import tarfile
import soundfile as sf
import io
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import json

class MfccPipeline:

    def __init__(self, PATH=None):
        if PATH is None:
            self.PATH = "C:/Users/alexc/Downloads/nsynth-train.jsonwav.tar.gz"
        else:
            self.PATH = PATH

        self.targets_list = [
            "bass", "brass", "flute", "guitar", "keyboard",
            "mallet", "organ", "reed", "string", "synth", "vocal"
        ]

    # def loop_all_files(self, PATH=None):
    #     if PATH is None:
    #         PATH = self.PATH
    #         # open the tar file
    #     with tarfile.open(PATH, 'r:gz') as tar:

    #         # Initialize counter to count number of files pulled.
    #         index = 0
    #         while index < 300000:
    #             fname = tar.next()
    #             if index % 1000 == 0:
    #                 print(index)
    #             index += 1
    # recover wav files from tarball
    def get_files(self, PATH=None, num_files = 100, verbose=False):
        """Sequentially pull wav files from .tar.gz file
        :param PATH: path to compressed dataset file
        :param num_files: number of files to read
        :returns: 4 generators for data (wav file converted to list), sr (sample rate)
                target name (e.g. 'guitar', 'mallet', ect.), and target index
        """

        if PATH is None:
            PATH = self.PATH

        # open the tar file
        with tarfile.open(PATH, 'r:gz') as tar:

            # Initialize counter to count number of files pulled.
            index = 0
            while index < num_files:
                fname = tar.next()

                # print output for logging
                if index % 1000 == 0:
                    print("\ntrain_data_preparation.py: looping through index: ", index, "\n")
                # Break if there are no more files.
                if fname is None:
                    break

                # Check that we're dealing with the proper format
                if fname.name.endswith(".wav"):
                    if verbose:
                        print(fname)

                    # Extract file
                    wav_file = tar.extractfile(fname).read()

                    # Convert bytes to a readable format
                    data, sr = sf.read(io.BytesIO(wav_file))

                    # Get target from filename
                    target = fname.name.split('/')[2].split('_')[0]

                    # yeild the 4 generators
                    try:
                        yield data, sr, target, self.targets_list.index(target)
                    except StopIteration:
                        return
                    index += 1

    # collect raw data into an array
    def get_dataset(self, num_files=10, num_batches=10, verbose=False):
        """Use the generator returned by get_files to append to the datset.
        This function itself will return a generator to get the next batch of data in the dataset.
        Example:
        `data_out = get_dataset(num_files=100, num_batches=10`
        calling next(dat_out) will return a dataset with the files 0-100.
        The second call to next(dat_out) will return a dataste with files 101-200
        """

        data_generator = self.get_files(self.PATH, num_files*num_batches, verbose)
        for i in range(num_batches):
            data = []
            for j in range(num_files):
                data.append(next(data_generator))
            X, t = self.prepare_data(data)
            #try:
            yield X, t
            #except StopIteration:
            #    return

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

    ### TODO: finish working on this function
    # def get_json(self, PATH=None,):
    #     if PATH is None:
    #         PATH = self.PATH

    #     # open the tar file
    #     with tarfile.open(PATH, 'r:gz') as tar:
    #         while index < num_files:
    #             fname = tar.next()

    #             # Break if there are no more files.
    #             if fname is None:
    #                 break

    #             # Check that we're dealing with the proper format
    #             if fname.name.endswith(".json"):
    #                 print(fname)
    #                 json_file = tar.extractfile(fname).read()

    def mfcc_pipeline(self, num_samples, num_batches, verbose=False):
        """Write this doc later"""

        # get the data from tar file
        data = self.get_dataset(num_samples, verbose)
        X, t = self.prepare_data(data)

        # split into train and validation set and return the data
        #return train_test_split(X, t, test_size=validation_split)

if __name__ == '__main__':
    pipe = MfccPipeline()
    test = pipe.get_dataset()

    something = next(test)
    print("features: ", something[0])
    print("targets: ", something[1])

    print("shape of features: ", something[0].shape)