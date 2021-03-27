from sqlalchemy import create_engine
#import pymysql
import pandas as pd

defaultCols = ['mfcc', 'filename', 'target', 'alternate_target']


def _read_table(tableName, columns=['*']):
    cnx = create_engine('mysql+pymysql://root:s11versouL@localhost/music')
    sqlFriendlyColumns = ', '.join(columns)
    df = pd.read_sql('Select %s From %s' %
                     (sqlFriendlyColumns, tableName.split()[0]), cnx)
    return df


def readTest():
    return _read_table('test', defaultCols)


def readTrain():
    return _read_table('train', defaultCols)


def readValidate():
    return _read_table('validate', defaultCols)


if __name__ == '__main__':
    df = _read_table('test', defaultCols)
    print(df)
