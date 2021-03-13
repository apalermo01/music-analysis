import pandas as pd
import mysql.connector


def getConnection():
    return mysql.connector.connect(
        user='root', password='s11versouL', database='music')


def add_1000(filenames, df):
    '''It doesn't really have to be 1000. \
    The dataframe length can be longer than the filenames length.'''
    cnx = getConnection()
    cursor = cnx.cursor()
    for file in filenames:
        try:
            insertTrain(cursor, file, df.loc[file[:-4]])
        except Exception as e:
            message = 'Could not retrieve %s from json file dataframe.' % e
            print(message)
    cnx.commit()
    cnx.close()


def insertTrain(cursor, filename, target, fft='', mfcc='', folder_id=1):
    '''Just insert training data into the train table.'''
    add_train = ("INSERT INTO train "
                 "(fft, mfcc, filename, folder_id, target) "
                 "VALUES (%(fft)s, %(mfcc)s, %(filename)s, %(folder_id)s, %(target)s)")
    data_train = {'filename': filename, 'target': target,
                  'fft': fft, 'mfcc': mfcc, 'folder_id': folder_id}
    err = 0
    exceptions = 0
    try:
        cursor.execute(add_train, data_train)
    except Exception as e:
        err += 1
        exceptions = e
    if err:
        print('errors "%s": %d' % (exceptions, err))
