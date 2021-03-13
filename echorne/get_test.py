from os import getcwd, walk, path
import pandas as pd
import add_test

audioFolder = path.join(getcwd(), 'audio')
_, _, musicFiles = next(walk(audioFolder))

examples = pd.DataFrame(pd.read_json(
    'examples.json').transpose().loc[:, 'instrument'])

for i in range((examples.shape[0] // 1000)+1):
    realIndex = i*1000
    maxIndex = (realIndex+1000) if realIndex + \
        1000 <= len(musicFiles) else len(musicFiles)
    musicFilesSlice = musicFiles[realIndex:maxIndex]
    add_test.add_1000(musicFilesSlice, examples)
