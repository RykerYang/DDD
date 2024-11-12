import os
import sqlite3

from src.main.utils.Path import getResourcePath


class SQLTool:
    def __init__(self, resource_path):
        # ______（________，_____）
        self.db_file = os.path.join(resource_path, 'sqlite3.db')

        # ___SQLite___
        self.conn = sqlite3.connect(self.db_file)
        # cursor = self.conn.cursor()

    def query(self, sql: str, params: tuple = None):
        cursor = self.conn.cursor()
        cursor.execute(sql, params)
        result = cursor.fetchall()
        cursor.close()
        return result

    def execute(self, sql: str, params: tuple = None):
        cursor = self.conn.cursor()
        cursor.execute(sql, params)
        self.conn.commit()
        cursor.close()

    def getConn(self):
        return self.conn

    def close(self):
        self.conn.close()


sqlTool = SQLTool(resource_path=getResourcePath())
