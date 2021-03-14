import numpy as np
import librosa


def getFftMfcc(wav):
    y, sr = librosa.load(wav, mono=True)
    fft = np.fft.fft(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    return str(fft), str(mfcc)
