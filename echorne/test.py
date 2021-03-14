import numpy as np
import librosa


def getFftMfcc(wav):
    y, sr = librosa.load(wav, mono=True)
    fft = np.fft.fft(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    fftStr = str(list(fft.round(0)))
    mfccStr = str(list(mfcc))
    return fftStr, mfccStr


with open('test.txt', 'w+') as test:
    fft, mfcc = getFftMfcc('audio/bass_electronic_018-022-100.wav')
    totalChars = test.write('Just the FFT\n\n' + fft)
    print('fft: %d chars long' % totalChars)
    totalChars = test.write('\n\nMFCC\n\n' + mfcc)
    print('mfcc: %d chars long' % totalChars)
