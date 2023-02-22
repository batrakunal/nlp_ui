"""
Author: Dario Borrelli
Date: July 2019
"""

import numpy as np
import pandas as pd
import re
import string
import glob
import pickle
import string
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import gzip
import urllib.request as urllib
from tqdm import tqdm
warnings.filterwarnings("ignore")
import wikipedia
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from string import punctuation
from stop_words import get_stop_words
import requests
from pycm import *
import urllib.request as urllib
from components.Text_cleaning import lower, spacing, eliminate_punctuation, eliminate_stopwords, text_cleaner



def import_conll2000(trainset=True, testset=False, merge=True):
    """
    This function will download the conll2000 dataset. 

        :param trainset: returns the trainset
        :type: bool
        :param testset: returns the testset
        :type: bool
        :param merge: returns the merge of train and test
        :type: bool

        :returns: the dataset in a pandas dataframe
        :rtype: pandas dataframe
    """

    url1 = "https://www.clips.uantwerpen.be/conll2000/chunking/train.txt.gz"
    url2 = "https://www.clips.uantwerpen.be/conll2000/chunking/test.txt.gz"

    

    urllib.urlretrieve(url1, "train.txt.gz")
    urllib.urlretrieve(url2, "test.txt.gz")

    file_test = gzip.open("test.txt.gz", "r")
    file_train = gzip.open("train.txt.gz", "r")

    trainset = True
    testset = True

    if trainset == True:

        tokens = []
        labels = []
        phrase_type = []

        for line in file_train:
            try:
                s = line.split()
                tokens.append(s[0].decode("utf-8"))
                labels.append(s[2].decode("utf-8").split("-")[0])

                try:
                    phrase_type.append(str(line).split("-")[1][:2])
                except:
                    phrase_type.append("O")

            except:
                pass
        conll = pd.DataFrame()
        conll["tokens"] = tokens
        conll["iob"] = labels
        conll["phrase"] = phrase_type

        conll_train = conll

        print(len(tokens), "tokens")

    if testset == True:

        tokens = []
        labels = []
        phrase_type = []

        for line in file_test:
            try:
                s = line.split()
                tokens.append(s[0].decode("utf-8"))
                labels.append(s[2].decode("utf-8").split("-")[0])

                try:
                    phrase_type.append(str(line).split("-")[1][:2])
                except:
                    phrase_type.append("O")

            except:
                pass

        conll = pd.DataFrame()
        conll["tokens"] = tokens
        conll["iob"] = labels
        conll["phrase"] = phrase_type
        conll_test = conll
        print(len(tokens), "tokens")

    if merge == True:
        try:

            merge = pd.concat([conll_train, conll_test], axis=0)

        except:
            print("works only if trainset and testset both True!")

    output = []

    try:
        output.append(conll_train)
    except:
        pass

    try:
        output.append(conll_test)
    except:
        pass

    try:
        output.append(merge)
    except:
        pass

    return output

def ngramming(text, n):

    """
    This function will return a n-grammed text.
    
    paramters:
    text: pass a cleaned text as a string.
    n: pass 1 for unigrams, 2 for bigrams, 3 for trigrams, 4 for fourgrams, 5 for fivegrams.
    """

    unigrams = text.split(" ")

    ngrams = []

    if n == 1:
        # print("uni-grammed")
        return unigrams

    if n == 2:
        # print("\n2-grams\n")
        for i in range(0, len(unigrams)):
            try:
                ngrams.append(unigrams[i] + " " + unigrams[i + 1])
            except:
                # print("bi-grammed!")
                pass

    if n == 3:
        # print("\n3-grams\n")
        for i in range(0, len(unigrams)):
            try:
                ngrams.append(
                    unigrams[i] + " " + unigrams[i + 1] + " " + unigrams[i + 2]
                )
            except:
                # print("tri-grammed!")
                pass
    if n == 4:
        # print("\n4-grams\n")
        for i in range(0, len(unigrams)):
            try:
                ngrams.append(
                    unigrams[i]
                    + " "
                    + unigrams[i + 1]
                    + " "
                    + unigrams[i + 2]
                    + " "
                    + unigrams[i + 3]
                )
            except:
                # print("four-grammed!")
                pass

    if n == 5:
        # print("\n5-grams\n")
        for i in range(0, len(unigrams)):

            try:
                ngrams.append(
                    unigrams[i]
                    + " "
                    + unigrams[i + 1]
                    + " "
                    + unigrams[i + 2]
                    + " "
                    + unigrams[i + 3]
                    + " "
                    + unigrams[i + 4]
                )
            except:
                # print("five-grammed!")
                pass
    if n == 6:
        # print("\n6-grams\n")
        for i in range(0, len(unigrams)):

            try:
                ngrams.append(
                    unigrams[i]
                    + " "
                    + unigrams[i + 1]
                    + " "
                    + unigrams[i + 2]
                    + " "
                    + unigrams[i + 3]
                    + " "
                    + unigrams[i + 4]
                    + " "
                    + unigrams[i + 5]
                )
            except:
                pass
    if n == 7:
        # print("\n7-grams\n")
        for i in range(0, len(unigrams)):

            try:
                ngrams.append(
                    unigrams[i]
                    + " "
                    + unigrams[i + 1]
                    + " "
                    + unigrams[i + 2]
                    + " "
                    + unigrams[i + 3]
                    + " "
                    + unigrams[i + 4]
                    + " "
                    + unigrams[i + 5]
                    + " "
                    + unigrams[i + 6]
                )
            except:
                pass

    if n == 8:
        # print("\n8-grams\n")
        for i in range(0, len(unigrams)):

            try:
                ngrams.append(
                    unigrams[i]
                    + " "
                    + unigrams[i + 1]
                    + " "
                    + unigrams[i + 2]
                    + " "
                    + unigrams[i + 3]
                    + " "
                    + unigrams[i + 4]
                    + " "
                    + unigrams[i + 5]
                    + " "
                    + unigrams[i + 6]
                    + " "
                    + unigrams[i + 7]
                )
            except:
                pass

    return ngrams


def cumulative(lists):

    """This function will calculate the cumulative of a list of numbers. It will be used in the ngrams_data function
    
    lists: pass a list of integers or digits.
    """

    cu_list = []
    length = len(lists)
    cu_list = [sum(lists[: x + 1]) for x in range(0, length)]
    return cu_list


def ngrams_data(text_clean, n):
    """
    This function will return a pandas dataframe containing information on frequency, percentage, and cumulative for a given ngrams list.
    
    parameters:
    text_clean: pass a text properly cleaned with basic_cleaning.
    n: pass the "n" of the n-grams you want to analyze.
    
    """

    ngrams = ngramming(text_clean, n)
    # print(ngrams)
    dn = freqs(ngrams)

    total_freq = sum(dn.values())

    dn_perc = {}

    for i in range(0, len(ngrams)):
        dn_perc[ngrams[i]] = dn[ngrams[i]] / total_freq

    df = pd.DataFrame(dn.values(), index=dn_perc.keys(), columns=["freq"])
    df["perc"] = dn_perc.values()
    df.sort_values(["perc"], ascending=True, inplace=True)

    df["cumul"] = cumulative(list(df["perc"]))

    return df


def freqs(ngrams):

    """
    This function will return the frequency of each ngram in a python dictionary, where keys are n-grams and values are frequencies.
    
    parameters:
    ngrams: just pass a list of ngrams.
    """

    countgrams = {}

    for i in range(0, len(ngrams)):

        cnt = ngrams.count(ngrams[i])

        countgrams[ngrams[i]] = cnt

    return countgrams


def total_tokens(text_clean):
    """
    Metric for computing the total number of tokens in the cleaned text.
    
    Parameters:
    text_clean: just pass a text cleaned with basic_cleaning.
    """
    return len(text_clean.split())


def unique_grams(data):
    """
    Metric for computing the number of unique ngrams.
    
    parameters:
    data: pass data stats (ngrams_data function output)
    
    """
    return len(data)


def repeated_counts(data):
    """
    This metric will count the number of times repeated ngrams will appear in the text (basically ngrams with frequency > 1).
    parameters:
    data: pass data stats (ngrams_data function output)
   
    """
    return len(data[data["freq"] > 1])


def freq_repeated_counts(data):
    """
    This metric will compute the sum of frequencies of ngrams with frequency > 1.
    
    parameters:
    data: pass data stats (ngrams_data function output)
    """
    df = data[data["freq"] > 1]
    return sum(list(df["freq"]))


def frequent_chunks(text_clean, start=2, end=5):
    """
    Frequent chunks identification.
        :param text_clean: pass a cleaned text
        :type: string
        :param start: starts with 2 (2-grams) or a different n-gram.
        :type: int
        :param end: ends with 5 (5-grams) or a different n-gram.
        :type: int 
    """
    to_be_chunked = []

    for i in range(start, end):

        data = ngrams_data(text_clean, i)

        for chunk in list(data[data["freq"] > 1].index):

            to_be_chunked.append(chunk)

    return to_be_chunked


def remove_boundaries(text, lang="en"):
    """This function will remove unwanted boundary words typical of n-gram frequency-based chunking. 

        :param text: pass a chunk as a string
        :type: string
        :param lang: pass the language (eg.: "en" for english, "it" for italian, "es" for spanish, default is english language)
        :type: string

        :returns: the string the unwanted boundary words
        :rtype: string
    """

    if lang == "en":

        # stops =  list(set(stopwords.words('english')))
        stops = get_stop_words("en")

        try:
            stops.remove("the")
        except:
            pass
        try:
            stops.remove("an")
        except:
            pass

    else:

        stops = get_stop_words(lang)

    l = text.split(" ")
    try:

        if l[0] in stops:
            l.remove(l[0])
        if l[-1] in stops:
            l.remove(l[-1])

    except:
        pass

    new = " ".join(l)

    return new


def replaced2bio(final):
    """
    This function will convert a final list of chunks into tokens + BIO labels (Begin-Inside-Outside) with a pandas dataframe type.
    
    parameters:
    
    final: pass a list of chunks.
    """

    t = []
    l = []

    for chunk in final:
        toks = chunk.split(" ")
        for i in toks:

            if any(j in i for j in punctuation) is True:
                t.append(i)
                l.append("O")
            else:
                if toks.index(i) == 0:
                    t.append(i)
                    l.append("B")
                else:
                    t.append(i)
                    l.append("I")

    final_iob = pd.DataFrame()
    final_iob["tokens"] = t
    final_iob["iob"] = l
    return final_iob


def clean_list(l, lang="en"):

    """This function will further clean a list of chunks.

        :param l: pass a list of chunks
        :type: list
        :param lang: pass the language (eg.: "en" for english)
        :type: string

        :returns: cleaned list 
        :rtype: list
    """
    #stops =  list(set(stopwords.words('english')))
    stops = get_stop_words("en")
    r = set([text_cleaner(i, lowercase=True, insert_spaces=False, remove_punctuation=True, remove_stopwords=True, stopwords=stops) for i in l])
    
    new1 = list(r)  

    for i in new1:
        i = i.replace("\n", "")
        leng = len(i.split(" "))
        if leng == 1:
            new1.remove(i)

    return new1


def load_alanritter():
    """
    This function will load the Alan Ritter Twitter annotated corpus with BIO labels.
    Place alanritter.txt in the working directory 
    (download it manually from here: https://github.com/aritter/twitter_nlp/blob/master/data/annotated/chunk.txt)
    """

    df = pd.read_csv(
        "chunk.txt",
        sep=" ",
        quotechar=None,
        quoting=3,
        header=None,
        names=["tokens", "iob"],
    )
    new_iob = []
    new_tokens = []

    for i in df.iob:
        try:

            s = i.split("-")
            new_iob.append(s[0].upper())
        except:
            new_iob.append(i.upper())

    for i in df.tokens:
        new_tokens.append(i.replace("_", ""))

    df_final = pd.DataFrame()
    df_final["tokens"] = new_tokens
    df_final["iob"] = new_iob

    return df_final


def createxy(text):
    """
    This function will create metrics on frequencies of chunks for a given text.
    
    parameters:
    text: just pass a text as a string.
    """
    tc = basic_cleaning(text)

    y = []
    x = []
    u = []
    for i in range(1, 8):
        df = ngrams_data(tc, i)
        x.append(i)
        y.append(max(df["freq"]))
        u.append(unique_grams(df))
        # print(i, unique_grams(df), repeated_counts(df), freq_repeated_counts(df), total_tokens(tc) - freq_repeated_counts(df) + repeated_counts(df) -1 * (i - 1), max(df['freq']))
    return x, y, u


def plot_one(x, y, u, leng, save=True):
    """Plots max frequency of n-grams.
    
    parameters:
    x: first value generated by createxy function.
    y: second value generated by createxy function.
    z: third value generated by createxy function.
    leng: length of the corpus (integer)
    save: set True if you want to save the figures (default is True).
    """

    fig, ax = plt.subplots()

    d = pd.DataFrame(data={"x": x, "y": y})
    plt.plot("x", "y", data=d, linestyle="--", marker="s", color="b")
    plt.xlabel("N")
    plt.ylabel("Max Frequency")
    # plt.xticks([1,1,7])
    # plt.yticks([1,100,10000])
    plt.ylim([1, 10000])

    axes = plt.gca()
    ax.tick_params(direction="in")

    a = plt.axes([0.58, 0.53, 0.28, 0.28], facecolor="w")

    d1 = pd.DataFrame(data={"x": x, "u": u})
    plt.plot("x", "u", data=d1, linestyle=":", marker="s", color="r")

    plt.title("Corpus Length = " + str(leng) + " tokens")
    # plt.xticks([1,1,7])
    plt.xlabel("N")
    plt.ylabel("Unique N-grams")

    plt.plot()
    plt.savefig(str(leng) + ".pdf")


def unsupervised_chunking(text, lang="en", replace=True, removepunct=False, removestopwords=False):
    """
    This method is a wrap of all previous functions for a smooth unsupervised chunking process with one line of code call.
    
    parameters:
    text: pass a text to be chunked as a string.
    lang: pass the language "en" is english and it is set as default.
    removepunct: pass True if you want to remove punctuations in the output list of chunks.
    removestopwords: pass True if you want to remove stopwords in the output list of chunks. 
    """

    print("\nperforming step 1: basic pre-process...\n")
    remstops = get_stop_words("en")
    #one = basic_process(text.lower())
    one = spacing(lower(text))
    print("performing step 2: frequent chunks count...\n")
    two = frequent_chunks(one)
    print("performing step 3: cleaning most frequent chunks list...\n")
    three = clean_list(two, lang=lang)
    print("performing step 4: producing final list...\n")
    
    if replace == True:
        
        return three
    
    if replace == False:
        
        if removepunct == False:

            end = chunk_replace(three, spacing(lower(text)))

            if removestopwords == True:

                end_clean = [x for x in end if x not in remstops]

                while "" in end_clean:
                    end_clean.remove("")

                return [i.replace("\n", " ").lstrip() for i in end_clean]

            if removestopwords == False:

                end_clean = end
                while "" in end_clean:
                    end_clean.remove("")
                return [i.replace("\n", " ").lstrip() for i in end_clean]

        if removepunct == True:

            end = chunk_replace(three, spacing(lower(text)))
            end_nop = [
                i
                for i in end
                if (list(set(i).intersection(string.punctuation)) == []) is True
            ]

            if removestopwords == True:

                end_clean = [x for x in end_nop if x not in remstops]
                while "" in end_clean:
                    end_clean.remove("")
                return [eliminate_punctuation(lower(i)) for i in end_clean]

            if removestopwords == False:

                end_clean = end_nop
                while "" in end_clean:
                    end_clean.remove("")
                return [eliminate_punctuation(lower(i)) for i in end_clean]


def save_chunks(filename, predict):
    """
    Save chunks in a pickle file.
    
    parameters:
    filename: pass the name of the file for example "chunk_list".
    predict: pass the list of chunks to be saved.
    """

    with open(filename + ".pickle", "wb") as file:
        pickle.dump(predict, file)
        print("file saved!")


def load_chunks(filename):
    """
    Load chunks from a pickle file.
    
    parameters:
    filename: pass the file name of the pickle file where the chunk list is stored.
    """

    with open(filename + ".pickle", "rb") as f:
        chunks_list = pickle.load(f)
        print("file loaded!")
        return chunks_list
