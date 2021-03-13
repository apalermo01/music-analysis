import mysql.connector


def getConnection():
    return mysql.connector.connect(
        user='root', password='s11versouL', database='music')
