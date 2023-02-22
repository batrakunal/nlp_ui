
"""
Authors:
Carlo Lipizzi, Dario Borrelli, Antonio Pugliese
Info at clipizzi@stevens.edu


This file will clean a txt file, extract bigrams and trigrams and generate a file with all the clean words.
Words/ngrams in the output file keep the same cardinality as they have in input.
The output file has 1 word/chunk per line.

"""

# importing the required libraries
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from datetime import datetime


def txt_clean(word_list, min_len):
    """
    Performs a first cleaning to a list of words.
    
    :param word_list: list of words
    :type: list
    :param min_len: minimum length of a word to be acceptable
    :type: integer
    :return: clean_words list of clean words
    :type: lists

    """
    clean_words = []
    for line in word_list:
        parts = line.strip().split()
        for word in parts:
            word_l = word.lower()
            if word_l.isalpha():
                if len(word_l) > min_len:
                    clean_words.append(word_l)

    return clean_words

def list_sublist (list_small, list_large):
    """
    Checks if a list an ngram is part of a (n+1)gram.
    If yes, it eliminated the ngram
    
    :param list_small: list of ngrams
    :type chunk_list: list
    :param list_large: list of (n+1)grams
    :type text: list
    :return: new_short_set, list_large revised lists of ngrams and (n+1)grams
    :type: lists

    """
    # this is to check if any element of a list of an ngrams is in the list of a (n+1)grams
    for ngram_small in list_small:
        if ngram_small in list_large:
            list_large.remove(ngram_small)
            #print ('\nremoved', ngram_small)
    
    # this is to remove ngrams when part of the list of (n+1)grams
    to_be_removed_lst = []
    for single_s in list_small:
        for single_l in list_large:
            if single_s in single_l:
                to_be_removed_lst.append(single_s)
                break

    new_short_set = list(set(list_small) - set(to_be_removed_lst))
    return new_short_set, list_large


def stopword_removal(word_list, stopwords_list):
    """
    Eliminates stopwords from a list of words.
    
    :param word_list: list of words
    :type chunk_list: list
    :param stopwords_list: list of stopwords
    :type text: list
    :return: clean_words, a list of words with no stopwords
    :type: list

    """
    clean_words = []
    for word in word_list:
        if word.strip() not in stopwords_list:
            clean_words.append(word)
        else:
            continue
            #print ('word', word, 'removed')
    return clean_words


def right_left_clean (ngram_list,stopwords_list):
    """
    Eliminates stopwords placed on the right or left side of an ngram.
    
    :param ngram_list: list of ngrams
    :type chunk_list: list
    :param stopwords_list: list of stopwords
    :type text: list
    :return: clean_ngram_list, a list of ngram with no stopwords on the right or left
    :type: list

    """
    if len(ngram_list) == 0:
        return ngram_list
    ngram_type = len(ngram_list[0].split())
    for num in range(1, ngram_type):
        clean_ngram_list = []
        for single_ngram in ngram_list:
            single_ngram_lst = single_ngram.split()
            first, *middle, last = single_ngram_lst
            if (first not in stopwords_list) and (last not in stopwords_list):
                clean_ngram_list.append(single_ngram)
            else:
                if len(single_ngram_lst) == 2:
                    #print ('removing for len = 2!')
                    continue
                if first in stopwords_list:
                    single_ngram_lst = single_ngram_lst [1:]
                    clean_ngram_list.append(' '.join(single_ngram_lst))
                    #print ('removing because of', first, 'new ngram =', single_ngram_lst)
                    continue
                if last in stopwords_list:
                    single_ngram_lst = single_ngram_lst [:-1]
                    clean_ngram_list.append(' '.join(single_ngram_lst))
                    #print ('removing because of', last, 'new ngram =', single_ngram_lst)
        ngram_list = clean_ngram_list
    return clean_ngram_list


def chunk_replacement(chunk_list, text):
    """
    Connects words chunks in a text by joining them with an underscore.
    
    :param chunk_list: word chunks
    :type chunk_list: list of strings/ngrams
    :param text: text
    :type text: string
    :return: text with underscored chunks
    :type: string

    """
    for chunk in chunk_list:
        text = text.replace(chunk, chunk.replace(' ', '_'))
    return text



def ngramming_bforce2 (word, stopwords, word_len, ngram_range, bigram_base_frequency, trigram_base_frequency):
    """
    This function is a frequency-based n-grammer.
    It also clean the right and left boundaries from stopwords.
    Returns a list of idiomatic expressions.
    
    :param word: string with the a corpus 
    :type word: string
    :param stopwords: list of stopwords
    :type stopwords: list
    :param ngram_range: tuple (min,max) for the n-gramming
    :type ngram_range: tuple
    :param ngram_base_frequency: value of a baseline frequency of ngrams in the text
    :type ngram_base_frequency: integer
    :return: list of words/ngrams
    :type: list
    :return: text_chunked_clean_lst, all_chunkslist of chunks lists of clean chunked text and chunks
    :type: lists

    The threshold for the ngrams is calculated as
        int(ngram_base_frequency*len(-text-))
        That makes the frequency a linear finction of the length of the document

    """
    
#    input_file = open(file,'r', encoding='utf8')
    
    # initializing list of input words
    in_file = []

    # populating the list of words from the text file
#
    if word != '\n':
        word_cln = txt_clean(word.split(' '), min_len = word_len)
        in_file.append(' '.join(word_cln))

    #print ('\n---Starting the Vectorization process')
    vectorizer = CountVectorizer(analyzer='word', stop_words=None, ngram_range=ngram_range)
    vec_fit = vectorizer.fit_transform(in_file)
    vec_fit_array = vec_fit.toarray()
    count_values = vec_fit_array.sum(axis=0)
    vocab = vectorizer.vocabulary_
    # the following is creating a list of tuples (frequency, word)
    counts = sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
    # print ('\n---The input files contains', len(counts), 'words')
    
    bigram_freq_calc = len(counts)**bigram_base_frequency
    trigram_freq_calc =len(counts)**trigram_base_frequency
    
    # print ('   Only bigrams occurring more than', bigram_freq_calc, 'times and')
    # print ('    trigrams occurring more than', trigram_freq_calc, 'times will be added to the list\n')

    df = pd.DataFrame(counts, columns =['Frequency', 'Name'])

    # the following is calculating the number of words in the ngrams in the data structure
    df['ngram_type'] = df['Name'].str.split().str.len()

    bigrams_list = df[(df.Name.str.split(' ').str.len() ==ngram_range[0]) &
        (df.Frequency >bigram_freq_calc)]['Name'].values.tolist()
    trigrams_list = df[(df.Name.str.split(' ').str.len() ==ngram_range[1]) &
        (df.Frequency >trigram_freq_calc)]['Name'].values.tolist()
    # print ('- Before cleaning there are', len(bigrams_list), 'bigrams and', len(trigrams_list), 'trigrams\n')
   
    clean_bigrams_list_tmp = right_left_clean (bigrams_list,stopwords)
    clean_trigrams_list_tmp = right_left_clean (trigrams_list,stopwords)
    
    clean_bigrams_list, clean_trigrams_list = list_sublist (clean_bigrams_list_tmp, clean_trigrams_list_tmp)

    # the following is merging the bigrams and trigrams in a single list
    all_chunks = clean_bigrams_list + clean_trigrams_list

    # the following is to replace single words parts of ngrams with the whole ngrams
    text_str = ' '.join(str(x) for x in in_file)
    text_chunked = chunk_replacement(all_chunks, text_str)

    # the following is transforming the string of text into a list and removing the stopwords
    text_chunked_lst = list(text_chunked.split(' '))
    text_chunked_clean_lst = stopword_removal(text_chunked_lst, stopwords)

    # printing the results and returning the chunked text and the ngrams
    # print ('- After cleaning there are', len(clean_bigrams_list), 'bigrams and', len(clean_trigrams_list), 'trigrams')
    # print ('    for a total of', len(text_chunked_clean_lst), 'words')

    return text_chunked_clean_lst, all_chunks


#----------- Main



# Parameters definition

text_file = '(Book I) Harry Potter and the Sorcerer_s Stone.txt'
stp_file = 'stopwords_en.txt'
word_min_len = 2
ngram_min = 2
ngram_max = 3
bigram_base_frequency = 0.28 #the higher the number, the less bigrams will be extracted
trigram_base_frequency = 0.28 #the higher the number, the less trigrams will be extracted
txt_out = 'txt_clean.txt'

# opening the stopword file
stopwords_file = open(stp_file,'r', encoding='utf8')

# initializing list of stopwords
stopwords = []

# populating the list of stopwords
for word in stopwords_file:
    stopwords.append(word.strip())

# printing starting time
start_time = datetime.now()
print ('\n-Starting the ngramming process at {}'.format(start_time), '\n')
#print('--- starting time: {}'.format(start_time))

# calling the ngramming function

text_final_lst, chunks = ngramming_bforce2 (text_file, stopwords, word_min_len,
    (ngram_min, ngram_max), bigram_base_frequency, trigram_base_frequency)

print ('the following are the resulting n-grams:\n', chunks)

# writing the results

with open(txt_out, mode='wt', encoding='utf-8') as out_file:
    out_file.write('\n'.join(str(line) for line in text_final_lst))

#print ('\nthose are the ngrams:', chunks)

# end of the process

print ('\n---End of the process. The total duration is {}'.format(datetime.now() - start_time), '\n')

#print('--- total duration: {}'.format(datetime.now() - start_time), '\n')


