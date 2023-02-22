"""
Complex components available:
- Embeddings matrices for google W2V model, specialized pdf model (Yan Dataset)
"""

import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from scipy.spatial import distance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import stop_words
from sklearn.metrics import euclidean_distances
from pyemd import emd
import numpy as np
import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from tqdm import tqdm
from datetime import datetime
import pymongo
from pytz import timezone
import spacy
from bs4 import BeautifulSoup as bs
import requests
from operator import itemgetter
import string
import re

from fake_useragent import UserAgent
import tensorflow as tf

from spacy import displacy
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from components.wrapperflowW2V import *


def generate_set(df):

    """
    This function will generate the set for the Local Outlier Factro for the prediction purpose of the incoming document
    Param df: Just pass the dataframe of the papers after generating the chunks. It will create a list and generate the model
    Output: dataframe with the chunks and vector value

    """

    chunks = []

    for i in range(len(df)):
        chunks.extend(df["chunks"][i])

    embedding_size = 10
    embedding_window = 5
    num_iterations = 1000
    model = WrapperFlow(chunks, embedding_size, embedding_window, num_iterations)
    model.run()
    embedding = model.embeds

    model_set = pd.DataFrame.from_dict(embedding)
    model_set = model_set.T
    return model_set


def LocalOutlierFactor_Prediction(train_set, test_set):

    """
    This function uses the localoutlierfactor method for the prediction purpose. 

    param train_set: Set which is generated from 'generate_set' function for training set
    param test_set: Set which is generated from 'generate_set' function for test set

    Output: List of the values with 1 and -1 from the test_set chunks and "1" indicates positive whereas "-1" indicates negative

    """
    lof = LocalOutlierFactor(novelty=True)
    lof.fit(train_set)
    lof_results = lof.predict(test_set)
    lof_result = lof_results.tolist()
    return lof_result
