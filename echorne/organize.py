import os
import re

os.chdir('/Volumes/SandiskUSB3/nsyth/train/audio')
for root, dirs, files in os.walk('.'):
    for wavFl in files:
        family = re.sub(r'^[^_]+_([^_]+)_[0-9].*', r'\1', wavFl)
        instrument = re.sub(r'^([^_]+)_.*', r'\1', wavFl)
        destFolder = os.path.join(family, instrument)
        destination = os.path.join(destFolder, wavFl)
        if not os.path.exists(destFolder):
            if not os.path.exists(family):
                os.mkdir(family)
            os.mkdir(destFolder)
        os.rename(wavFl, destination)
        
