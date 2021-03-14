import numpy as np
import librosa


def getFftMfcc(wav):
    y, sr = librosa.load(wav, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfccStr = str(list(mfcc))
    return mfccStr
