import mysql.connector

cnx = mysql.connector.connect(
    user='root', password='s11versouL', database='music')
cursor = cnx.cursor()

add_train = ("INSERT INTO train "
             "(filename, target) "
             "VALUES (%(filename)s, %(target)s)")
data_train = {'filename': 'testing', 'target': 333}

cursor.execute(add_train, data_train)
t_no = cursor.lastrowid
print(t_no)
cnx.commit()

cnx.close()
