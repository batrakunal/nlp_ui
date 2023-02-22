import psycopg2
import pandas.io.sql as psql
import pandas as pd
from sqlalchemy import create_engine


def query_postgre_database(query="",
                            database = "postgres", 
                            host = "wrt1010.sercuarc.org",
                            port="27018", 
                            user = "postgres",
                            password = 'example456'):

    """
    This function will retrieve data tables from the postgre database, 
    and shape them into a pandas dataframe ready to be used in python.
    
    Parameters:
    
    query: pass an SQL SELECT query for example "SELECT * FROM data_gathering.monitoring_news"
    database: pass a string with the name of the database to connect for example "postgres" (as default)
    host: pass a string with the url of the hosting server.
    port: pass a string with the port number to connect to.
    user: pass a string with the name of the user of the postgre database.
    password: pass a string with the password of the postgre database.
    
    """

    conn = psycopg2.connect(host = host,
                            database = database, 
                            user = user, 
                            port = port, 
                            password = password)

    df = psql.read_sql(query, con = conn)
    
    return df

  
def write_dataframe_postgreSQL(df, table_name, connection_string):
  
    """
    This function will write a pandas dataframe into our postgreSQL database.

    Parameters:

    df: pass a pandas dataframe to be stored in the db.
    table_name: pass a string with the name of table you want to save.
    connection_strin: pass a string containing the information for connecting to the postgreSQL.
                      eg. 'postgresql://scott:tiger@localhost:5432/mydatabase'
    """
    
    engine = create_engine(connection_string)
    df.to_sql(table_name, engine) 
    
    return print("table written in postgreSQL database.")
