import pandas as pd
import mysql.connector
import fft_mfcc
from os import path

TABLE = 'test'

def getConnection():
    return mysql.connector.connect(
        user='root', password='s11versouL', database='music')


def add_1000(filenames, df, relativePath=''):
    '''It doesn't really have to be 1000. \
    The dataframe length can be longer than the filenames length.'''
    cnx = getConnection()
    cursor = cnx.cursor()
    for filename in filenames:
        wav = filename
        if relativePath:
            wav = path.join(relativePath, filename)
        try:
            mfcc = fft_mfcc.getFftMfcc(wav)
            insertTest(cursor, filename, df.loc[filename[:-4]], None, mfcc)
        except Exception as e:
            message = 'Could not retrieve %s from json file dataframe.' % e
            print(message)
    cnx.commit()
    cnx.close()


def insertTest(cursor, filename, target, fft='', mfcc='', folder_id=1):
    '''Just insert testing data into the test table.'''
    add_test = (f"INSERT INTO {TABLE} "
                 "(fft, mfcc, filename, folder_id, target) "
                 "VALUES (%(fft)s, %(mfcc)s, %(filename)s, %(folder_id)s, %(target)s)")
    data_test = {'filename': filename, 'target': target,
                  'fft': fft, 'mfcc': mfcc, 'folder_id': folder_id}
    err = 0
    exceptions = 0
    try:
        cursor.execute(add_test, data_test)
    except Exception as e:
        err += 1
        exceptions = e
    if err:
        print('errors "%s": %d' % (exceptions, err))