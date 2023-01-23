import logging
import psycopg2

logging.basicConfig(level=logging.DEBUG, filename='db_connection.log')


def create_db_connection(db):
    conn = None
    try:
    	conn = psycopg2.connect(host="localhost",
                                port=5432,
                                user="postgres",
                                password="123456",
                                database=db
                                )
    except psycopg2.Error as e:
        
        logging.error("Error while connecting to PostgreSQL: %s", e)
        print("Error while connecting to PostgreSQL", e)
        
    else:
        logging.info("Connected to %s.", db)
        print(f"Connect at {db}")
    
    return conn

def sql_command(sql, cursor):
    cursor.execute(sql)    
    return
