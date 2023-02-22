
"""
Authors:
Carlo Lipizzi, Dario Borrelli, Antonio Pugliese
Info at clipizzi@stevens.edu


    
This file will clean a txt file, extract bigrams and trigrams and generate a file with all the clean words.
Words/ngrams in the output file keep the same cardinality as they have in input.
The output file has 1 word/chunk per line.

updated by:
    Hojat Behrooz 03/31/2022

this version fixed the minor problem which ignore the words concatinated to puntucations
such as 'he went out, while he was out of city.' in this sample 'out' was ignored.
the new version fix this problem and improve the performance also it keeps the orginal
structure by adding space instead of any ignoring characters and the words. 
it helps to keep track the cleaned text be relted to the orginal positioning of the
corpus. it helps to find the orginal text postion by knowing the cleaned postion.
"""

# importing the required libraries
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from datetime import datetime
import numpy as np
import re



def stopword_removal(txt, stopwords_list):
    """
    Eliminates stopwords from a list of words.
    
    :param word_list: list of words
    :type chunk_list: list
    :param stopwords_list: list of stopwords
    :type text: list
    :return: clean_words, a list of words with no stopwords
    :type: list

    """
    for word in stopwords_list:
       txt= re.sub(" "+word+" ", (len(word)+2)*' ', txt)

            #print ('word', word, 'removed')
    return txt


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
    cln_ngram=[]
    for item in ngram_list:
        not_clean=True
        single_ngram=item.split()
        while(not_clean):
            if(len(single_ngram)<2):
                not_clean=False
            elif(single_ngram[0] in stopwords_list):
                single_ngram =single_ngram[1:]
            elif(single_ngram[len(single_ngram)-1] in stopwords_list):
                single_ngram =single_ngram[0:-1]
            else:
                new_gram = ' '.join(single_ngram)
                if(len(cln_ngram)!=0):
                    if (new_gram not in cln_ngram) & (np.sum([new_gram in x for x in cln_ngram])==0):
                        cln_ngram.append(new_gram)
                else:
                    cln_ngram.append(new_gram)
                    
                not_clean=False
    return(cln_ngram)



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
        pattern= " "+chunk.replace(' ', '( +)')+" "
        for m in re.finditer(pattern,text):
            text=text[:m.start(0)]+" "+str(chunk.replace(' ', '_')+(-1-len(chunk)-m.start(0)+m.end(0))*' ')+text[m.end(0):]
    return text

def word_len_limit(text,word_len):
    """
    
    replace words with length less than wod_len from the text with space
    Parameters
    ----------
    text : string
        the raw text.
    word_len : integer 
        the minimum length of words.

    Returns
    -------
    the cleaned text.

    """
    start=-1
    for ind in range(len(text)):
        if(text[ind]==' ') & (start!=-1):
            if(ind-start<word_len):
                text=text[:start]+(ind-start)*" "+text[ind:]
            start=-1
        elif (text[ind]!=' ') & (start==-1):
            start=ind
    return(text)
        
                


def ngramming_bforce2 (rawText, stopwords, word_len, ngram_range, bigram_base_frequency, trigram_base_frequency):
    """
    This function is a frequency-based n-grammer.
    It also clean the right and left boundaries from stopwords.
    Returns a list of idiomatic expressions.
    
    :param rawText: string contians the raw orginal text 
    :type rawText: string
    :param stopwords: list of stopwords
    :type stopwords: list
    :param ngram_range: tuple (min,max) for the n-gramming
    :type ngram_range: tuple
    :param ngram_base_frequency: value of a baseline frequency of ngrams in the text
    :type ngram_base_frequency: integer
    :return: string of cleaned text contins words/ngrams
    :type: string
    :return: text_chunked_clean_lst, all_chunkslist of chunks lists of clean chunked text and chunks
    :type: lists

    The threshold for the ngrams is calculated as
        int(ngram_base_frequency*len(-text-))
        That makes the frequency a linear finction of the length of the document

    """
    #add aspace on starting and end of raw text
    rawText=" "+rawText+" "
    # Keep only alphabet and change to lower case
    in_file =re.sub('[^a-zA-Z]', ' ', rawText).lower()
    
    # ignore the words with len less than minumum word_len
    in_file= word_len_limit(in_file,word_len)
    #print ('\n---Starting the Vectorization process')
    vectorizer = CountVectorizer(analyzer='word', stop_words=None, ngram_range=ngram_range)
    vec_fit = vectorizer.fit_transform([in_file])
    vec_fit_array = vec_fit.toarray()
    count_values = vec_fit_array.sum(axis=0)
    vocab = vectorizer.vocabulary_
    # the following is creating a list of tuples (frequency, word)
    counts = sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
    print ('\n---The input files contains', len(counts), 'words')
    
    bigram_freq_calc = len(counts)**bigram_base_frequency
    trigram_freq_calc =len(counts)**trigram_base_frequency
    # print ('   Only bigrams occurring more than', bigram_freq_calc, 'times and')
    # print ('    trigrams occurring more than', trigram_freq_calc, 'times will be added to the list\n')

    df = pd.DataFrame(counts, columns =['Frequency', 'Name'])

    # the following is calculating the number of words in the ngrams in the data structure
    df['ngram_type'] = df['Name'].str.split().str.len()
    # find the ngrams which has frequency more than the minimums
    ngrams_list = df[((df.Name.str.split(' ').str.len() ==ngram_range[0]) &
                       (df.Frequency >bigram_freq_calc)) |
                      ((df.Name.str.split(' ').str.len() ==ngram_range[1]) &
                       (df.Frequency >trigram_freq_calc))   
                     ]['Name'].values.tolist()

#    print ('- Before cleaning there are', len(ngrams_list))
    # ignore ngrams which include stopwords in lef and right side
    clean_ngrams_list = right_left_clean (ngrams_list,stopwords)
    #replace ngrams with the equivalent with _ between words
    text_chunked = chunk_replacement(clean_ngrams_list, in_file)

    #remove stopwords 
    text_chunked_clean_lst = stopword_removal(text_chunked, stopwords)
#    print ('    for a total of', len(text_chunked_clean_lst), 'words')

    #remove the added spce in starting and ending of orginal text
    text_chunked_clean_lst = text_chunked_clean_lst[1:-1] #remove added blanks in fist place
    return text_chunked_clean_lst, clean_ngrams_list
"""
----------- Main



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

#read the file  content
with open(text_file, 'r') as filehandle:
    rawText = filehandle.read()
        
# calling the ngramming function to clean and ngram the text

text_final_lst, chunks = ngramming_bforce2 (rawText, stopwords, word_min_len,
    (ngram_min, ngram_max), bigram_base_frequency, trigram_base_frequency)

print ('the following are the resulting n-grams:\n', chunks)

# writing the results

with open(txt_out, mode='wt', encoding='utf-8') as out_file:
    out_file.write(text_final_lst)

#print ('\nthose are the ngrams:', chunks)

# end of the process

print ('\n---End of the process. The total duration is {}'.format(datetime.now() - start_time), '\n')

#print('--- total duration: {}'.format(datetime.now() - start_time), '\n')

"""
