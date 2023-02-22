from pymongo import MongoClient
import pandas as pd

def _connect_mongo(host, port, username, password, db):
    """ A util for making a connection to mongo """
    if username and password:
        conn = MongoClient(host=host, 
                     port=port,
                     username=username,
                     password=password)
    else:
        conn = MongoClient(host, port)
    return conn[db]


def read_mongo(db, collection, host, port, username, password, query={}, no_id=True):
    """ Read from Mongo and Store into DataFrame """
    # Connect to MongoDB
    db = _connect_mongo(host=host, port=port, username=username, password=password, db=db)
    # Make a query to the specific DB and Collection
    cursor = db[collection].find(query)
    # Expand the cursor and construct the DataFrame
    df =  pd.DataFrame(list(cursor))
    # Delete the _id
    if no_id:
        del df['_id']
    return df
