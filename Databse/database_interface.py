
import mysql.connector

class MySqlInterface:
    def __init__(self, host, user, password, database=None):
        self.host = host
        self.user = user
        self.password = password
        self.myserv = self.connect_server()
        if database is not None:
            self.mydb = self.connect_database(database)

    def connect_server(self):
        self.myserv = mysql.connector.connect(
            host=self.,
            user=self.user,
            password=self.password)
        return self.myserv

    def connect_database(self, database_name):
        self.mydb = mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=database_name)
        return self.mydb

    def create_database(self, database_name):
        try:
            mycursor = self.mydb.cursor()
            sql = "CREATE DATABASE " + database_name
            mycursor.execute(sql)
        except mysql.connector.Error as error:
            print("Failed to create table {}".format(error))

    def create_table(self, tab_name, db_dict):
        try:
            str_create_table = "CREATE TABLE " + tab_name + " ("
            string = ""
            for field in list(db_dict.keys()):
                string += field
                for keyword in db_dict[field]:
                    string += " " + keyword
                string += ", "
            string = string[:-2]
            string = string + ")"
            mycursor = self.mydb.cursor()
            mycursor.execute(str_create_table + string)
        except mysql.connector.Error as error:
            print("Failed to create table {}".format(error))


    def table_insert(self, table_name, data_dict, data_type):
        try:
            init_string = "INSERT INTO " + table_name + " ("
            feature_string = ""
            for feature in data_dict.keys():
                feature_string += feature + ", "
            feature_string = feature_string[:-2]
            value_string = ""
            for value in data_dict.values():
                value_string += str(value) + ", "
            value_string = value_string[:-2]
            sql_query = init_string + feature_string + ") VALUES (" + value_string + ")"
            cursor = self.mydb.cursor()
            cursor.execute(sql_query)
            self.mydb.commit()
            print(cursor.rowcount, "record inserted.")
            cursor.close()

        except mysql.connector.Error as error:
            print("Failed to insert record into Laptop table {}".format(error))

    def delete_database(self, database_name):
        try:
            mycursor = self.mydb.cursor()
            sql = "DROP DATABASE IF EXISTS " + database_name
            mycursor.execute(sql)
        except mysql.connector.Error as error:
            print("Failed to drop database {}".format(error))

    def delete_table(self, table_name):
        try:
            mycursor = self.mydb.cursor()
            sql = "DROP TABLE IF EXISTS " + table_name
            mycursor.execute(sql)
        except mysql.connector.Error as error:
            print("Failed to drop table {}".format(error))

    def show_databases(self):
        mycursor = self.mydb.cursor()
        mycursor.execute("SHOW DATABASES")
        for x in mycursor:
            print(x)

    def show_tables(self):
        try:
            cursor = self.mydb.cursor()
            cursor.execute("SHOW TABLES")
            for table_name in cursor:
                print(table_name)
        except mysql.connector.Error as error:
            print("Could not show tables {}".format(error))

    def show_columns(self, table_name):
        try:
            cursor = self.mydb.cursor()
            cursor.execute("SHOW columns FROM " + table_name)
            print([column[0] for column in cursor.fetchall()])
        except mysql.connector.Error as error:
            print("Could not show columns {}".format(error))


