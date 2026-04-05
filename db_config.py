import pyodbc

def get_connection():
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=Butsee\SQLEXPRESS;'
        'DATABASE=BrainTumorDetection;'
        'Trusted_Connection=yes;'
    )
    return conn