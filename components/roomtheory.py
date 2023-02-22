import numpy as np
from numpy import float32 as REAL
from gensim import matutils
from pymongo import MongoClient
import gridfs
import datetime
from bson.objectid import ObjectId
from scipy.spatial import distance
import pickle
from components.Chunking import chunking_soa, np_chunking
import pandas as pd
import psycopg2
import pandas as pd
import pickle
import time
from tqdm import tqdm
from components.Vectorization import tfidf
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
for word in ['us', ]:
    stopwords.add(word)
from components.Text_cleaning import *
import seaborn as sns
import matplotlib.pyplot as plt
from app import *

class Room(object):
    """
    Class that will handle the room attributes and methods
    *** This class it is not responsible for training, or preprocessing the data in any way, it will only take care of
    the post-processed data
    """
    def __init__(self, room_name, embedding, parameters):
        """
        Class constructor
        """
        assert type(room_name) == str
        assert type(parameters) == dict
        # Tag for the room
        self.room_name = room_name
        # Loads up a list of all the titles used in the room, together with its obj_id
        self._list_of_raw_obj_ids = []
        # Loads the embedding, from the already trained model
        self.embedding = embedding
        # Loads the parameters for creating of the room
        self._parameters = parameters

    @staticmethod
    def calculate_cosine_distance_str_w2v(astring, embeddings):
        """

        :param astring:
        :param embeddings:
        :param stopwords_after_chunking:
        :param remove_website:
        :param max_ngrams:
        :param min_count:
        :return:
        """
        # list_of_words = _create_chunks(astring, stopwords_after_chunking, remove_website, max_ngrams, min_count)
        list_of_words = astring.lower().split()
        list_vecs = []
        for word in list_of_words:
            try:
                list_vecs.append(embeddings[word])
            except KeyError:
                print(str(word) + ' - Not found')
        # TODO current implementation it is only using average, next step it is to solve this!
        matrix_vecs = np.asanyarray(list_vecs)
        ave_matrix_vecs = matrix_vecs.mean(0)
        ave_embeddings = np.asanyarray(list(embeddings.values())).mean(0)
        return 1 - distance.cosine(ave_matrix_vecs, ave_embeddings)

    def calculate_distance(self, astring):
        """
        :param astring:
        :return:
        """

        assert type(astring) == str
        if self.embedding is not None:
            return self.calculate_cosine_distance_str_w2v(astring.lower(), self.embedding)
        else:
            print('Load an embedding, either from pickle file or from mongodb.')
            return None

    def words_similarity(self, word1, word2):
        """

        :param word1:
        :param word2:
        :return:
        """
        if self.embedding is not None:
            return 1 - distance.cosine(self.embedding[word1], self.embedding[word2])
        else:
            return None

    def most_similar_wordlist(self, positive=None, negative=None, topn=10):
        """Find the top-N most similar words.
        Positive words contribute positively towards the similarity, negative words negatively.
        This method computes cosine similarity between a simple mean of the projection
        weight vectors of the given words and the vectors for each word in the model.
        The method corresponds to the `word-analogy` and `distance` scripts in the original
        word2vec implementation.
        Parameters
        ----------
        positive : list of str, optional
            List of words that contribute positively.
        negative : list of str, optional
            List of words that contribute negatively.
        topn : int, optional
            Number of top-N similar words to return.
        Returns
        -------
        list of (str, float)
            Sequence of (word, similarity).
        """
        if positive is None:
            positive = []
        if negative is None:
            negative = []

        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        positive = [(x, 1.0) for x in positive]
        negative = [(x, 1.0) for x in negative]

        # compute the weighted average of all words
        all_words, mean = set(), []
        for word, weight in positive + negative:
            mean.append(weight * self.embedding[word])
            if word in self.embedding.keys():
                all_words.add(word)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(np.array(mean).mean(axis=0)).astype(REAL)

        limited = np.array(list(self.embedding.values()))
        dists = np.dot(limited, mean)
        best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)
        # ignore (don't return) words from the input
        vocab = list(self.embedding.keys())
        best = [vocab[x] for x in best if vocab[x] not in all_words]
        return best[:topn]

    def __create_room_on_mongodb(self):
        client = MongoClient('wrt1010.sercuarc.org', 27017)
        tag = self.tag.replace(' ', '_')
        tag = 'ROOM_' + tag
        db = client[tag]

        client.close()

    def most_similar_vector(self, positive=None, negative=None, topn=10):
        """Find the top-N most similar words.
        Positive words contribute positively towards the similarity, negative words negatively.
        This method computes cosine similarity between a simple mean of the projection
        weight vectors of the given words and the vectors for each word in the model.
        The method corresponds to the `word-analogy` and `distance` scripts in the original
        word2vec implementation.
        Parameters
        ----------
        positive : list of str, optional
            List of words that contribute positively.
        negative : list of str, optional
            List of words that contribute negatively.
        topn : int, optional
            Number of top-N similar words to return.
        Returns
        -------
        list of (str, float)
            Sequence of (word, similarity).
        """
        if positive is None:
            positive = []
        if negative is None:
            negative = []

        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        positive = [(x, 1.0) for x in positive]
        negative = [(x, 1.0) for x in negative]

        # compute the weighted average of all words
        all_words, mean = set(), []
        for word_vec, weight in positive + negative:
            mean.append(weight * word_vec)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(np.array(mean).mean(axis=0)).astype(REAL)

        limited = np.array(list(self.embedding.values()))
        dists = np.dot(limited, mean)
        best = matutils.argsort(dists, topn=topn, reverse=True)
        # ignore (don't return) words from the input
        vocab = list(self.embedding.keys())
        best = [vocab[x] for x in best]
        return best[:topn]


class RoomTheory(object):
    """
    Class to identify rooms
    """

    def __init__(self, threshold, connection_string, list_of_rooms=None):
        """
        Class constructor
        :param threshold: Defines the minimum distance to be considered the same room [-1.0, 1.0] interval
        :param load_room_method:
        :param prepare_for_offline_access:
        """
        self.rooms = {}
        if list_of_rooms is not None:
            for r in list_of_rooms:
                room = load_room(r, connection_string)
                self.rooms[room.room_name] = room
        else:
            avai_rooms = get_available_rooms(connection_string)
            for r in avai_rooms:
                aroom = load_room(r, connection_string)
                self.rooms[aroom.room_name] = aroom
        self.threshold = threshold

    def __words_similarity_by_room(self, word1, word2, room_tag):
        """

        :param word1:
        :param word2:
        :return:
        """
        return self.rooms[room_tag].words_similarity(word1, word2)

    def room_distance(self, room_tag, astring):
        """

        :param astring:
        :return:
        """
        return self.rooms[room_tag].calculate_distance(astring)

    def get_distance_all_rooms(self, astring):
        """

        :param astring:
        :return:
        """
        rtn = {}
        for k in self.rooms.keys():
            rtn[k] = self.room_distance(self.rooms[k], astring)
        return rtn

    def distance_between_rooms(self, room_tag1, room_tag2):
        """
        :return:
        """
        romm1_ave = np.array(list(self.rooms[room_tag1].embedding.values())).mean(0)
        romm2_ave = np.array(list(self.rooms[room_tag2].embedding.values())).mean(0)
        return 1 - distance.cosine(romm1_ave, romm2_ave)

    def get_impact_factor(self, room_tag, list_of_benchmark_words):
        """
        :param list_of_benchmark_words:
        :return:
        """
        rtn_dict = {}
        for word in list_of_benchmark_words:
            rtn_dict[word] = self.rooms[room_tag].calculate_distance(word)
        return rtn_dict

    @staticmethod
    def softmax_normalizer(alist_of_numbers):
        """
        :param alist_of_numebrs:
        :return:
        """
        anarray = np.array(alist_of_numbers)
        exp_array = np.exp(anarray)
        summ = exp_array.sum()
        return list(exp_array / summ)

    def get_room_view_benchmark(self, room_tag, list_of_benchmark_words, new_document_as_string, cross_ref_benchmark=[],
                                calculate_impact_factor=False, verbose=False, normalize=False, distribution_metrics=False, 
                                spatial_distance="cosine", chunking_method='np'):
        """
        :param list_of_benchmark_words:
        :param cross_ref_benchmark:
        :param new_document_as_string:
        :param calculate_impact_factor:
        :param verbose:
        :return:
        """
        list_of_benchmark_words = [x.lower() for x in list_of_benchmark_words]
        cross_ref_benchmark = [x.lower() for x in cross_ref_benchmark]
        room = self.rooms[room_tag]
        if calculate_impact_factor:
            dict_of_benchmark_words_impactfactor = self.get_impact_factor(room_tag, list_of_benchmark_words)
        else:
            dict_of_benchmark_words_impactfactor = {key: 1.0 for key in list_of_benchmark_words}

        bench_matrix = {}
        for bench in list(dict_of_benchmark_words_impactfactor.keys()):
            try:
                bench_matrix[bench] = room.embedding[bench]
            except KeyError:
                if verbose:
                    print('Skipped %s', bench)

        cross_bench_matrix = {}
        for bench in cross_ref_benchmark:
            try:
                cross_bench_matrix[bench] = room.embedding[bench]
            except KeyError:
                if verbose:
                    print('Skipped %s', bench)

        if chunking_method == 'soa':
            alist_of_words = chunking_soa(new_document_as_string.lower())
        elif chunking_method == 'np':
            alist_of_words = np_chunking(new_document_as_string.lower())
        else:
            alist_of_words = new_document_as_string.lower().split()

        new_doc_matrix = {}
        for word in alist_of_words:
            try:
                new_doc_matrix[word] = room.embedding[word]
            except KeyError:
                if verbose:
                    print('Skipped %s', word)

        return_matrix = {}
        for k, v in new_doc_matrix.items():
            return_matrix[k] = []
            for key, value in bench_matrix.items():
                if spatial_distance == "cosine":           
                    return_matrix[k].append(1 - distance.cosine(v, value) * dict_of_benchmark_words_impactfactor[key])
                if spatial_distance == "1-cosine":           
                    return_matrix[k].append(distance.cosine(v, value) * dict_of_benchmark_words_impactfactor[key])
                if spatial_distance == "euclidean":           
                    return_matrix[k].append(distance.euclidean(v, value) * dict_of_benchmark_words_impactfactor[key])

        cross_return_matrix = {}
        for k, v in new_doc_matrix.items():
            cross_return_matrix[k] = []
            for key, value in cross_bench_matrix.items():
                if spatial_distance == "cosine":    
                    cross_return_matrix[k].append(1 - distance.cosine(v, value))
                if spatial_distance == "1-cosine":
                    cross_return_matrix[k].append(distance.cosine(v, value))
                if spatial_distance == "euclidean":
                    cross_return_matrix[k].append(distance.euclidean(v, value))
                    

        df_cross = pd.DataFrame(data=cross_return_matrix)
        df = pd.DataFrame(data=return_matrix)      
        df.index = list(bench_matrix.keys())

        df_pre = df
        
        #summing the distances, multiplied by the
        df = df.sum(axis=1)
        df_cross = df_cross.sum(axis=1)
            
        df = pd.DataFrame(df)
        df_cross = pd.DataFrame(df_cross)
        #normalize each line to
        if df_cross.shape[0] != 0:
            rtn_df = pd.DataFrame(data=np.dot(df, df_cross.T), columns=cross_ref_benchmark, index=df.index)
            df_series = rtn_df.apply(self.softmax_normalizer, axis=1)
        elif normalize is True:
            rtn_df = df
            df_series = rtn_df.apply(self.softmax_normalizer, axis=0)
        else:
            rtn_df = df
            df_series = rtn_df
        cols_to_drop = rtn_df.columns
        if normalize is True:
            for i in range(0, rtn_df.shape[1]):
                col_name = str(rtn_df.columns[i]) + '_norm'
                rtn_df[col_name] = df_series.apply(lambda x: x[i], axis=1)
            # rtn_df.drop(cols_to_drop, axis=1, inplace=True)
        # rtn_df['distance'] = df.values
        
        #taking care of distribution of intermediate scores
        if distribution_metrics == True:
            
            df = df_pre.sum(axis=1)
            df_cross = df_cross.sum(axis=1)
            d = {}
            for i in list(df_pre.index):

                d[i] = dict(df_pre.loc[i].describe())

            distr_out = pd.DataFrame(d).T
            df_dist = pd.concat([rtn_df, distr_out], axis=1)
            return df_dist
        else: 
            
            return rtn_df


def create_room(room_name, embeddings, parameters, connection_string):
    """
    Create a room and loads it onto the MongoDB
    :param room_name: name of the room you want to create
    :type room_name: string
    :param embeddings: pass the corresponding embeddings
    :type embeddings: dict
    :param parameters: additional parameters
    :type parameters:
    :param connection_string: the mongodb connection string 
    :type connection_string:
    :return: Room
    :rtype: roomtheory.Room
    """
    client = MongoClient(connection_string)
    
    tag = room_name.replace(' ', '_')
    tag = 'ROOM_' + tag
    db = client['HOTEL']
    collection_info = db['info']
    fs = gridfs.GridFS(db)
    pickle_file = pickle.dumps(embeddings)
    obj_id = fs.put(pickle_file)
    insert_dict = parameters
    insert_dict['_id'] = tag
    insert_dict['embedding_id'] = str(obj_id)
    insert_dict['timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    collection_info.insert_one(insert_dict)
    client.close()
    return Room(room_name, embeddings, parameters)


def get_available_rooms(connection_string):
    """
    :param connection_string: the mongodb connection string 
    :return: a room created in the mongodb.
    """
    
    # MONGO_CLIENT = MongoClient(connection_string)
    # db = MONGO_CLIENT['admin']
    collection_info = admindb['fs.files']
    available_rooms = [str(id) for id in collection_info.find().distinct('_id')]
    return available_rooms


def load_room(room_name, connection_string):
    """
    :param room_name: pass the room name of the room to be loaded.
    :param connection_string: the mongodb connection string 
    """
    
    client = MongoClient(connection_string)
    
    db = client['admin']
    collection_info = db['fs.files']
    available_rooms = [str(id) for id in collection_info.find().distinct('_id')]
    if room_name not in available_rooms:
        print("Room not available, please check spelling or use the function `get_available_rooms()` to get a "
              "list of available rooms")
        return None
    a = collection_info.find_one({'_id': room_name})
    embedding_id = a['embedding_id']
    delete = a.pop('timestamp', None)
    delete = a.pop('_id', None)
    delete = a.pop('embedding_id', None)
    fs = gridfs.GridFS(db)
    file = fs.get(ObjectId(embedding_id))
    binary = file.read()
    embedding = read_pickle_obj(binary)
    return Room(room_name.replace('ROOM_', ''), embedding, a)


def save_to_pickle(name, object):
    """

    :param name:
    :param object:
    :return:
    """
    try:
        file = open(str(name)+'.pickle', 'ab')
        pickle.dump(object, file)
        file.close()
        return True
    except:
        return False


def read_pickle_file(filename):
    """

    :param filename:
    :return:
    """
    try:
        with open(filename, "rb") as input_file:
            e = pickle.load(input_file)
        return e
    except:
        return False


def read_pickle_obj(binary_obj):
    """

    :param filename:
    :return:
    """
    try:
        e = pickle.loads(binary_obj)
        return e
    except:
        return False

    
class RoomTheory_val(object):
    """
    Class to identify rooms
    """

    def __init__(self, threshold, user, password, list_of_rooms=None):
        """
        Class constructor
        :param threshold: Defines the minimum distance to be considered the same room [-1.0, 1.0] interval
        :param load_room_method:
        :param prepare_for_offline_access:
        """
        self.rooms = {}
        if list_of_rooms is not None:
            for r in list_of_rooms:
                room = load_valid_room(r,user=user,password=password)
                self.rooms[room.room_name] = room
        else:
            avai_rooms = get_available_valid_rooms(user=user,password=password)
            for r in avai_rooms:
                aroom = load_valid_room(r,user=user,password=password)
                self.rooms[aroom.room_name] = aroom
        self.threshold = threshold

    def __words_similarity_by_room(self, word1, word2, room_tag):
        """

        :param word1:
        :param word2:
        :return:
        """
        return self.rooms[room_tag].words_similarity(word1, word2)

    def room_distance(self, room_tag, astring):
        """

        :param astring:
        :return:
        """
        return self.rooms[room_tag].calculate_distance(astring)

    def get_distance_all_rooms(self, astring):
        """

        :param astring:
        :return:
        """
        rtn = {}
        for k in self.rooms.keys():
            rtn[k] = self.room_distance(self.rooms[k], astring)
        return rtn

    def distance_between_rooms(self, room_tag1, room_tag2):
        """

        :return:
        """
        romm1_ave = np.array(list(self.rooms[room_tag1].embedding.values())).mean(0)
        romm2_ave = np.array(list(self.rooms[room_tag2].embedding.values())).mean(0)
        return 1 - distance.cosine(romm1_ave, romm2_ave)

    def get_impact_factor(self, room_tag, list_of_benchmark_words):
        """

        :param list_of_benchmark_words:
        :return:
        """
        rtn_dict = {}
        for word in list_of_benchmark_words:
            rtn_dict[word] = self.rooms[room_tag].calculate_distance(word)
        return rtn_dict

    @staticmethod
    def softmax_normalizer(alist_of_numbers):
        """

        :param alist_of_numebrs:
        :return:
        """
        anarray = np.array(alist_of_numbers)
        exp_array = np.exp(anarray)
        summ = exp_array.sum()
        return list(exp_array / summ)

    def get_room_view_benchmark(self, room_tag, list_of_benchmark_words, new_document_as_string, cross_ref_benchmark=[],
                                calculate_impact_factor=False, verbose=False, normalize=False, distribution_metrics=False,
                                spatial_distance="cosine", chunking_method='np'):
        """

        :param list_of_benchmark_words:
        :param cross_ref_benchmark:
        :param new_document_as_string:
        :param calculate_impact_factor:
        :param verbose:
        :return:
        """
        list_of_benchmark_words = [x.lower() for x in list_of_benchmark_words]
        cross_ref_benchmark = [x.lower() for x in cross_ref_benchmark]
        room = self.rooms[room_tag]
        if calculate_impact_factor:
            dict_of_benchmark_words_impactfactor = self.get_impact_factor(room_tag, list_of_benchmark_words)
        else:
            dict_of_benchmark_words_impactfactor = {key: 1.0 for key in list_of_benchmark_words}

        bench_matrix = {}
        for bench in list(dict_of_benchmark_words_impactfactor.keys()):
            try:
                bench_matrix[bench] = room.embedding[bench]
            except KeyError:
                if verbose:
                    print('Skipped %s', bench)

        cross_bench_matrix = {}
        for bench in cross_ref_benchmark:
            try:
                cross_bench_matrix[bench] = room.embedding[bench]
            except KeyError:
                if verbose:
                    print('Skipped %s', bench)

        if chunking_method == 'soa':
            alist_of_words = chunking_soa(new_document_as_string.lower())
        elif chunking_method == 'np':
            alist_of_words = np_chunking(new_document_as_string.lower())
            chunk_rep = chunk_replacement(alist_of_words, new_document_as_string.lower()).split(" ")
            alist_of_words = [i.replace("_"," ") for i in chunk_rep if i not in stopwords]
        else:
            print("No chunking defined")
            alist_of_words = new_document_as_string.lower().split()

        new_doc_matrix = {}
        for word in alist_of_words:
            try:
                new_doc_matrix[word] = room.embedding[word]
            except KeyError:
                if verbose:
                    print('Skipped %s', word)
              
        return_matrix = {}
        for k, v in new_doc_matrix.items():
            return_matrix[k] = []
            for key, value in bench_matrix.items():
                if spatial_distance == "cosine":           
                    return_matrix[k].append(1 - distance.cosine(v, value) * dict_of_benchmark_words_impactfactor[key])
                if spatial_distance == "1-cosine":           
                    return_matrix[k].append(distance.cosine(v, value) * dict_of_benchmark_words_impactfactor[key])
                if spatial_distance == "euclidean":           
                    return_matrix[k].append(distance.euclidean(v, value) * dict_of_benchmark_words_impactfactor[key])
        
        cross_return_matrix = {}
        for k, v in new_doc_matrix.items():
            cross_return_matrix[k] = []
            for key, value in cross_bench_matrix.items():
                if spatial_distance == "cosine":    
                    cross_return_matrix[k].append(1 - distance.cosine(v, value))
                if spatial_distance == "1-cosine":
                    cross_return_matrix[k].append(distance.cosine(v, value))
                if spatial_distance == "euclidean":
                    cross_return_matrix[k].append(distance.euclidean(v, value))
                          
        
        df_cross = pd.DataFrame(data=cross_return_matrix)
        df = pd.DataFrame(data=return_matrix)  
        df.index = list(bench_matrix.keys())
        
        df_pre = df
        
        # summing the distances, multiplied by the
        df = df.sum(axis=1)
        df_cross = df_cross.sum(axis=1)
        
        df = pd.DataFrame(df)
        df_cross = pd.DataFrame(df_cross)
        # normalize each line to
        if df_cross.shape[0] != 0:
            rtn_df = pd.DataFrame(data=np.dot(df, df_cross.T), columns=cross_ref_benchmark, index=df.index)
            df_series = rtn_df.apply(self.softmax_normalizer, axis=1)
        elif normalize is True:
            rtn_df = df
            df_series = rtn_df.apply(self.softmax_normalizer, axis=0)
        else:
            rtn_df = df
            df_series = rtn_df
        cols_to_drop = rtn_df.columns
        if normalize is True:
            for i in range(0, rtn_df.shape[1]):
                col_name = str(rtn_df.columns[i]) + '_norm'
                rtn_df[col_name] = df_series.apply(lambda x: x[i], axis=1)
            # rtn_df.drop(cols_to_drop, axis=1, inplace=True)
        # rtn_df['distance'] = df.values
        
        
        #taking care of distribution of intermediate scores
        if distribution_metrics == True:
            df = df_pre.sum(axis=1)
            df_cross = df_cross.sum(axis=1)
            d = {}
            for i in list(df_pre.index):

                d[i] = dict(df_pre.loc[i].describe())

            distr_out = pd.DataFrame(d).T
            df_dist = pd.concat([rtn_df, distr_out], axis=1)          
            df_dist["mean"] /= df_dist["mean"].max()
            return df_dist
            
        return rtn_df
    
        
def write_rooms_in_postegre(room_name, pickle_file, host, database, user, port, password):
    
    """
    This function will write the pickle file of the rooms in the postgreSQL.
    
    Parameters:
    room_name: pass the name of the room as a string
    pickle_file: pass the pickles created with "pickle.dumps(roomobject)"
    """
    
    conn = psycopg2.connect(host=host,
                        database=database, 
                        user=user, 
                        port=port, 
                        password = password)
    
    cursor = conn.cursor()
    
    pick = pickle_file
    embedding_id = room_name + str(time.time())
    
    sql_insert = "INSERT INTO hotel.rooms_files (room_name, embedding_id, pickle) VALUES (%s, %s, %s);"
    
    cursor.execute(sql_insert, (room_name, embedding_id, pick))
    
    conn.commit()
    
def get_available_valid_rooms(host="wrt1010.sercuarc.org", 
                            database="rooms", 
                            user="",
                            port="27018", 
                            password=""):
    """
    This function will return all the available rooms in the postgreSQL database.
    """
    conn = psycopg2.connect(host=host,
                        database=database, 
                        user=user, 
                        port=port, 
                        password=password)
    
    cursor = conn.cursor()
    
    cursor.execute("""SELECT room_name FROM hotel.rooms_files""")
    
    rooms_av = []
    for table in cursor.fetchall():
        
        rooms_av.append(table[0])
    return rooms_av
    
def load_valid_room(room_name,user,password, 
                    host="wrt1010.sercuarc.org", 
                    database="rooms", 
                    port="27018"):    
    """
    This function will load a room from the postgreSQL.
    
    Parameters:
    room_name: pass the room_name as a string (to get the available rooms use get_available_rooms)
    """
    conn = psycopg2.connect(host=host,
                        database=database, 
                        user=user, 
                        port=port, 
                        password=password)
    
    cursor = conn.cursor()
    
    sql_query = "SELECT pickle FROM hotel.rooms_files WHERE room_name =" + "'" + room_name + "';"
    cursor.execute(sql_query)
    fetched = cursor.fetchall()
    print(fetched[0][0])
    
    return pickle.loads(fetched[0][0])


def docs_to_evaluate(docs, custom_keys=[]):
    
    keys = []
    if custom_keys == []:
        for i in range(0,len(docs)):
            keys.append("doc"+str(i))
    if custom_keys != []:
        if len(custom_keys) != len(docs):
            print("custom keys length must match docs length!")
        else:          
            keys = custom_keys
        
    incoming_docs = dict(zip(keys,docs))
    total_lens = sum([len(i) for i in docs])

    w = []
    for i in incoming_docs.values():
        w.append(len(i)/total_lens)
    #print(w)
    size_weights_docs = dict(zip(incoming_docs.keys(), w))
    return size_weights_docs,incoming_docs


def rooms_view_benchmark(room_theory_init,
                          rooms, 
                          docs, 
                          keywords,
                          custom_size_weights=[],
                          custom_doc_keys=[],
                          spatial_distance="cosine",
                          chunking_method='np',
                          verbose=False,
                          normalize=True,
                          distribution_metrics=False):  
    
    """
    This function calculates benchmark room theory scores for multiple rooms and multple documents.
    
    :param room_theory_init: room theory initilization object.
    :param rooms: a list of room tags e.g. ["ROOM_gps"]
    :type rooms: list
    :param docs: a list of documents to be evaluated via room theory
    :type docs: list
    :param keywords: list of benchmark keywords.
    :type keywords: list
    :return: dataframe with scores.
    :rtype: pandas dataframe
    """
    
    outputs = []
    results = {}

    weights,incoming_docs = docs_to_evaluate(docs,custom_keys=custom_doc_keys)
    
    if custom_size_weights != []:
        if len(custom_size_weights) != len(docs):
            print("custom_docs_weights and docs require same length!")
        else:       
            weights=dict(zip(incoming_docs.keys(),custom_size_weights))
    
    for doc in tqdm(incoming_docs.keys()):

        for room in rooms:

            bench_temp = [i for i in keywords if i != room.replace("ROOM_","").replace("_"," ")]

            try:
                res = room_theory_init.get_room_view_benchmark(room_tag=room.replace("ROOM_",""),
                                                list_of_benchmark_words=keywords,
                                                new_document_as_string=incoming_docs[doc],
                                                cross_ref_benchmark=[], # leave blank
                                                calculate_impact_factor=False, # set false
                                                verbose=False,
                                                normalize=normalize,
                                                spatial_distance=spatial_distance,
                                                chunking_method=chunking_method,
                                                distribution_metrics=distribution_metrics)
                res = res*weights[doc]
                res["room"] = [room.replace("ROOM_","")]*len(res)
                res["Text"] = [doc]*len(res)
                results[room] = res

            except:
                print(room.replace("ROOM_",""), "NaN")
        try:            
            if normalize == False:
                dfs = list(results.values())
                output = pd.concat([i.reset_index() for i in dfs], axis=0, ignore_index=True)
                output.columns = ['Benchmark','Score','Count','weighted_mean','std','min','25%','50%','75%','max','Room','Text']
                outputs.append(output)

            if normalize == True:
                dfs = list(results.values())
                output = pd.concat([i.reset_index() for i in dfs], axis=0, ignore_index=True)
                output.columns =['Benchmark','Score','Norm_Score','Count','weighted_mean','std','min','25%','50%','75%','max','Room','Text']
                outputs.append(output)
        except:
            print("no match found!")
    
    try:   
        outs = pd.concat(outputs)
        return outs
    except:
        print("no match found!")
        return None
    
def get_dynamics(outs,point_of_view,bench,metric,palette="prism"):
    """
    Plot dynamics of benchmark scores per each room.
    
    :param outs: the output of the function "rooms_view_benchmark()"
    :type outs: pandas dataframe
    :param point_of_view: list of tags (strings) of the rooms
    :type point_of_view: list
    :param bench: list of benchmark keywords
    :type bench: list
    :param palette: a palette or list of colors by matplotlib.
    :type palette: list
    :return: dataframe of dynamic scores (print charts as well).
    :rtype: pandas dataframe
    """
    
    dfs_tot = []    
    for b in bench:
        dfs = []
        for p in point_of_view:
            
            df1 = outs[(outs["Room"] == p)&(outs["Benchmark"] == b)].cumsum().drop(["Benchmark","Room","Text"], axis=1).reset_index().drop(["index"],axis=1).reset_index()
            df1["Room"] = [p]*len(df1)
            dfs.append(df1)
        df = pd.concat(dfs,axis=0)
        g = sns.FacetGrid(df, height=3, hue = "Room", palette=palette, legend_out=True)
    
        g.map(plt.plot,"index", metric, marker=".", markersize=12, markeredgecolor="w").set_axis_labels("index (narrative offset)", "compounded valence")
        plt.legend(frameon=True,edgecolor="black",framealpha=0.5)

        plt.title(b)
        plt.show()
        df["Benchmark"] = [b]*len(df)
        dfs_tot.append(df)
        
    final_df = pd.concat(dfs_tot,axis=0)
    
    return final_df
