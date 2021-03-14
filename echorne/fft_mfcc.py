import numpy as np
import librosa


def getFftMfcc(wav):
    y, sr = librosa.load(wav, mono=True)
    fft = np.fft.fft(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    fftStr = np.array2string(
        fft, precision=2, separator=',', suppress_small=True)
    mfccStr = np.array2string(
        mfcc, precision=2, separator=',', suppress_small=True)
    return fftStr, mfccStr
