# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 09:53:03 2022


This application take a text file and extract sentences from that
then cluster the sentense based on their centroied word2vec vector presntation


@author: Hojat
"""

from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')

filterwarnings(action='ignore', category=DeprecationWarning, message='`np.int` is a deprecated alias')
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.object` is a deprecated alias')
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.flo   at` is a deprecated alias')

import seaborn as sns
from Carlo_ngrams_tool.chunking_bforce_plus_space_add import ngramming_bforce2
from sklearn_extra.cluster import KMedoids
from io import BytesIO

import spacy
import numpy as np
import pandas as pd

import fitz
import pathlib

import os

from datetime import datetime
from gensim.models import Word2Vec


import nltk
nltk.download('stopwords')
nltk.download('wordnet')
sns.set_theme()


#from collections import Counter


sns.set_theme()
# conda install -c conda-forge scikit-learn-extra
# installing the extra
# load the pretrained Spacy model
nlp = spacy.load("en_core_web_lg")
# for downloading the model following must be done
# conda install -c conda-forge spacy-model-en_core_web_sm
# conda install -c conda-forge spacy-model-en_core_web_lg

# optimal number of cluste estimated by fiding the point with max distance from the line
# between first and last point on wcss
# thanks: https://jtemporal.com/kmeans-and-elbow-method/


#normalize a vector
def extractAll(user_file_path):
    def normalise(A):
        lengths = (A**2).sum(axis=1, keepdims=True)**.5
        return A/lengths

    def optimal_number_of_clusters(wcss):
        """
        The function take a list of wcss from kmean various # of cluster from 2 to max value 
        and find the point on the curve with the maximum distance to the line between
        first and last point of the curve as best # of cluster

        Parameters
        ----------
        wcss : List of the wcss value for the various kmean # of clusters  

        Returns
        -------
        int
            The optimal # of cluster .

        """
    #    coordination of the line between the first and last wcss points
        x1, y1 = 2, wcss[0]
        x2, y2 = len(wcss), wcss[len(wcss)-1]

        distances = []
        for i in range(len(wcss)):
            x0 = i+2
            y0 = wcss[i]
            numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
            denominator = ((y2 - y1)**2 + (x2 - x1)**2)**.5
            distances.append(numerator/denominator)

        return distances.index(max(distances)) + 2


    """
    https://reposhub.com/python/natural-language-processing/boudinfl-pke.html
    conda install -c conda-forge spacy-model-en_core_web_sm
    """


    def find_sentences(corpus_str):
        """
        find sentenses boudaries by utilizing Spacy library

        Parameters
        ----------
        corpus_str : string
            contins the entire input text.

        Returns
        -------
        a list of starting position of each sentence in the input text
        it allso add a pointer to the len of text as the last element 
        of list.

        """
        doc = nlp(corpus_str)
        nlp.max_length = 1500000
        # extract the sentences from the by looping over the processed file
        # it also extract the centroid vector of each sentence
        sentsp = []
        sentsv = []
        sents_start = []
        for sentp in doc.sents:
            sentsp.append(str(sentp.text))  # list of sentences
            sentsv.append(sentp.vector)    # list of the centroid sentence vector

        # iterating over the senetences and find the startpoint of the sentences in the source file
        # it seem the sentences  dose not match the exact indices related to the source corpus
        #

        start = 0
        sents_start = []
        for kk in range(len(sentsp)):
            index = corpus_str.find(sentsp[kk], start, len(corpus_str))
            start = index+len(sentsp[kk])
            sents_start.append(index)
        sents_start.append(len(corpus_str))
        return(sents_start)


    def extract_text_from_folders_of_pdfs(PDF_DATA_DIR, stopwords1,
                                        max_files=100, bench=None, doc_min_len=10, OrgDoc=False):
        """
        extracting files from PDF_DTA_DIR and clean and apply bigram and trigram
        it also use a benchmark to keep those words in dictionary dispite
        rare use in input files. it returns clean text from each file in a dataframe 
        format. if the  input parapeter OrgDoc=True then the ouput dataframe will be contined
        orginal text file,cleaned one, a list of starting index of pages, starting sentence
        index list, and a pointer to PyMUpdf text page pointer


        Parameters
        ----------
        PDF_DATA_DIR : directory contains pdf files
            DESCRIPTION.
        stopwords1 : stop words list
            DESCRIPTION.
        max_files : maximum files to read
            DESCRIPTION. The default is 100.
        bench : list of string
            the string list of benchmarks. The default is None.
        doc_min_len : define a minumim len for input file 
                    default is 10
        OrgDoc : Bolean with default value False define if the return dataframe
                would be included more detailes about input text file
        Returns
        -------
        rtn_dict : a dictionary of cleaned text which key is file name

        """
        if(OrgDoc):
            orgdoc_df = pd.DataFrame(
                columns=['file_name', 'org_doc', 'doc', 'page_start_list', 'sentence_start','fitz_doc'])
        else:
            orgdoc_df = pd.DataFrame(columns=['file_name', 'doc'])

        i = 0

        for(root, dirs, files) in os.walk(PDF_DATA_DIR, topdown=True):
            print("Inside Forrrrrr")
            for fname in files:
                if (fname.endswith('.pdf')): #only pdf will processed but it can process word and txt file as well
                    try:
                        fpath = os.path.join(root, fname)
                        doc = fitz.open(fpath)  # open document
                        rawText = ""  # open text output
                        page_start = []  # list of starting point of each page
    #                    sentnce_start_list
                        # set to ignore warning and error messge from mupdf
                        fitz.TOOLS.mupdf_display_errors(False)
                        for page in doc:  # iterate the document pages
                            # get plain text (is in UTF-8)
                            text = page.get_text("text", sort=True)
                            if(text != ""):
                                # add the starting point of the new page to page list
                                page_start.append(len(rawText))
                                rawText = rawText+str(text)  # write text of page
                                # print("RawText",rawText)
                        # check if the lenght of text is at least doc_min_len charachter
                        if(len(rawText) > doc_min_len):
                            txt, _ = ngramming_bforce2(rawText, stopwords1,
                                                    word_len=2, ngram_range=(2, 3),
                                                    bigram_base_frequency=.28,
                                                    trigram_base_frequency=.28)
                            if(OrgDoc):
                                orgdoc_df.loc[len(orgdoc_df)] = [
                                    fname, rawText, txt, page_start, find_sentences(rawText),doc]
                            else:
                                orgdoc_df.loc[len(orgdoc_df)] = [
                                    fname, txt.split()]

                            print(i, '+', fname, "processed")
                            i += 1
                            if(i == max_files):
                                return orgdoc_df.set_index('file_name')
                        else:
                            print(
                                fname, "documnet is too short, not processed", len(rawText))

                    except Exception as e:
                        print(e, '-', fname, ': pdf is not readable')
                        pass

        return orgdoc_df.set_index('file_name')


    def creat_bench(file_bench):
        """
        return a list of benchmarsk in file_bench

        Parameters
        ----------
        file_bench : string
            input file name.
        stopwords : list of strings
            list of stopwords.
        simple : True/False, optional 
            return simple combination or not. The default is True.

        Returns
        -------
        benchmark numpy list of string.
        weight of each benchmark

        """
        df = pd.read_csv(file_bench)
        st = []
        wt = []
        columns_words = [0]+list(range(3, len(df.columns)))
        for k in range(len(df)):
            for j in columns_words:
                if(df.iloc[k, j] == df.iloc[k, j]):
                    js = '_'.join(df.iloc[k, j].lower().split(' '))
                    if(js not in st):
                        st.append(js)
                        wt.append(df['weight'][k])
        return(st, wt)


    """
    -----main
    """

    """
    parapters used in the program
    """
    # Defining the path to the trainig pdf files
    folder_path = 'c:/dau'
    # folders for input files to be evaluated
    # proj_dir = './Docs with Acquisition Recommendations'
    proj_dir = user_file_path
    print(proj_dir)
    # benchmarks file
    file_bench = 'Bench_weighted_rooted.csv'
    # stop words file
    stp_file = 'stopwords_en.txt'

    # names for saving and loading the trained model
    model_file = 'word2vec_v1.model'
    #csv file which is created for presenting the similarity degree for each file in 
    #proj_dir to the benchmark
    sim_to_bench_file='sim_to_bench.csv'
    # maximum number of fils that has been read for training
    max_num_files = 2000

    word_min_len = 2
    ngram_min = 2  # minimum ngraming
    ngram_max = 3  # maximum ngraming

    # maximum # of cluster which is used to estimate the data clustring wcss
    max_cluster = 50


    # the text length would be exponentiation to -base_frequency to calculate minimum
    # frequency of the ngrams to be considered.
    # the higher the number, the less bigrams will be extracted
    bigram_base_frequency = 0.28
    # the higher the number, the less trigrams will be extracted
    trigram_base_frequency = 0.28

    # this treshhold is very important factor that select the sentences with highier 
    #weighted value compare to the tershold. total passed thershold words are divided by total 
    #number of doc words to calculate level of relation to the benchmark
    # highier value limit the number of sentence and vice versa
    similarity_tershold = .32


    ######################################
    #initilization
    stopwords_file = open(stp_file, 'r', encoding='utf8')

    # initializing list of stopwords
    stopwords1 = []

    # populating the list of stopwords
    for word in stopwords_file:
        stopwords1.append(word.strip())


    print('\n-Starting the pdf to txt conversion process-\n')
    start_time = datetime.now()
    print('--- starting time: {}'.format(start_time))


    # creat benchmark list from the input file
    bench, bench_w = creat_bench(file_bench)
    #%% this part only execute once for creating the model 
    #************very  processing expensive
    """
    # extracting text from all .pdf files from the specified folder path



    doc_df = extract_text_from_folders_of_pdfs(folder_path,
                stopwords1 ,max_num_files,bench=bench, OrgDoc=False)
    print ('\n--- this is the end of the process ---\n')

    print('--- total duration: {}'.format(datetime.now() - start_time))


    # trainig gensim word2vec model

    docs=list(doc_df.doc.values)

    print ("number of processed docs:",len(docs))

    cores = multiprocessing.cpu_count()

    t = time()
    w2vec_model = Word2Vec(min_count=10,
                        window=7,
                        vector_size=300,
                        sample=6e-5, 
                        alpha=0.03, 
                        min_alpha=0.0007, 
                        negative=20,
                        workers=cores)



    w2vec_model.build_vocab(docs, progress_per=10000)

    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
    t = time()

    w2vec_model.train(docs, total_examples=w2vec_model.corpus_count, 
                    epochs=50, report_delay=1.0)


    w2vec_model.save(model_file)
    print('Time to develop and train the model: {} mins'.format(round((time() - t) / 60, 4)))

    # test the trained model before saving
    print('most similar to recommendation:',
        w2vec_model.wv.most_similar(positive=['recommendation'],topn=10))

    """
    # %%
    # loading the trained model and creat word vocabulary
    # and collect the frequency of the words in vocab to calculate
    # rarness factor present as content measure
    w2vec = Word2Vec.load(model_file)
    w2vec_model = w2vec
    """ creating a vocaboulary dataframe with frequency"""
    vocab = list(w2vec.wv.key_to_index.keys())
    # creaat a vector numpy array fro mthe vocabulary

    print("the length of vocabolary is:", len(vocab))

    # creat a word dataframe from vocabulary including their frequency and length
    cnt = []
    for item in vocab:
        cnt.append(w2vec.wv.get_vecattr(item, "count"))
    voc_cnt = pd.DataFrame({'word': vocab, 'count': cnt})
    voc_cnt['len'] = voc_cnt['word'].apply(len)

    # calculate total input words to model after preprocess
    tot_cnt = np.sum(voc_cnt['count'])

    print("from total %d words %d unique words are added to vocabulary" %
        (tot_cnt, len(vocab)))

    # calculate content measure from the word frequency in vocabulary inspired by:
    # (Duong et al. 2019) it present the rareness of a word which make it more valuable
    # in a sentence to be concentrated as a key word
    #
    voc_cnt['content'] = [-np.log(x/tot_cnt) for x in voc_cnt['count']]
    voc_cnt['content'] /= voc_cnt['content'].max()
    # set the word as key for accesing vocabulary contents
    voc_cnt.set_index('word', inplace=True)

    # creat a df from the benchmarks and their weights
    bench_df = pd.DataFrame({'benchmark': bench, 'weight': bench_w})
    bench_df = pd.merge(left=bench_df, right=voc_cnt, left_on='benchmark', right_on='word')[
        ['benchmark', 'weight', 'count', 'content']]

    # caculate number of benchmarks that are not preseted in vocabulary
    bench_dif = list(set(bench)-set(vocab))
    bench_found = list(set(bench).intersection(set(vocab)))
    print("\n\n----Number of unknown words from bench in vocabulary:", len(bench_dif),
        "from:", len(set(bench)))
    print("------unknown words from bench which are ignored:\n", bench_dif)
    # ignore benchmarks wich is not in vocabulary
    bench_w = [bench_w[i] for i in range(len(bench)) if bench[i] in bench_found]
    bench = [bench[i] for i in range(len(bench)) if bench[i] in bench_found]
    # normalized bench marks weight between minimum and 1.
    bench_w /= np.max(bench_w)
    # pd.DataFrame(bench_dif).to_csv('Benchmarks_missing.csv')

    # create vocabulry df from vector, len,and count
    # voc_df = voc_cnt.copy()
    # voc_df[["d%03d" % (i) for i in range(300)]] = w2vec.wv[vocab]
    # # creat a sample vocabulary by count>50 and 20<len<25

    # sample_vocab = voc_df[(voc_df['count'] > 50) & (
    #     voc_df['len'] > 20) & (voc_df['len'] < 25)]

    proj_df = extract_text_from_folders_of_pdfs(proj_dir,
                                                stopwords1, bench=bench, OrgDoc=True)

    # %%
    # this part load a corpus and its sentences boundaries 
    # it use that to calculate the weighted average of each sentences vector
    # for weight the content factor of each word in vocabulary is used
    # this means a rare word is more important than a popular word for calcualting
    # the average. the weighted average would be feed to a kmediods model and find the
    # optimal number of cluster by applying elbow technic. the sentences in corpus will
    # clsuterd by the best number of cluster to find the related sentences in curpus.



    def cluster_sentences(corpus_str,cleaned_str,sents_start):
        """
        

        Parameters
        ----------
        corpus_str : string
            the input uncleaned corpus.
        cleaned_str : TYPE
            the input cleaned corpus.
        sents_start : TYPE
            list of the starting indice of corpus sentences.

        Returns
        -------
        clusterd_df_kmedoids: dataframe
                    a dataframe contines uncleaned snetnces and their cluster label
        
        clusterd_df_stats: data frame
                    a dataframe contines cluster labels and the mediods sentences of
                    each cluster

        """    
    # this section iterate over sentences in corpus and  find the word2vec presentation of existing
    # word in sentences then caclualte average weighted vector by using content as weight of
    # each word   . for sentences with no words in vocab , no vector is recorded.

        sentsp_pure = []  # list of sentences that have at least one word in vocab
        sents_list = []   # list of index of the sentences in orginal sentences list
        # list of the average weighted word2vec vector for the sentence
        sents_weighted_vector = []
        for ind in range(len(sents_start)-1):
            start = sents_start[ind]
            end = sents_start[ind+1]
            focus_sents = cleaned_str[start:end].split(' ')
        
            # select the words from documnets which have entery in vocab
            sub_sent = list(set(focus_sents).intersection(set(vocab)))
            # document words vectors  multiply by the content measure
            # which present the rarreness and calculate the average of the weighted sentences vectors
            if(len(sub_sent) != 0):
                w_vec = np.sum(w2vec_model.wv[sub_sent].T * voc_cnt.loc[sub_sent].content.values,
                            axis=1) / voc_cnt.loc[sub_sent].content.sum()
                sents_weighted_vector.append(w_vec)
                sents_list.append(ind)
                sentsp_pure.append(cleaned_str[start:end])
        
        # creat an numpy array from the weighted centroid vector of all sentences
        X = np.array(sents_weighted_vector)
        
        
        ################ cluster the sentences
        
        # this part applies Kmedoids clustering method to find the nearest pointd to each other
        # in a form of distance matrix. distnace matrix has been produced from the cosine similarity
        # measure. the xdot is cosine similarity matrix between all words in document
        # I belive as the k-means use the euclidain distance and use average distance
        # it is not good measure for comparing word2vec vectors. a better approch would be using
        # cosine simialrity matrix and apply  Kmedoids as technic
        # I have transfer the cosine similarity to distance by subtracting it form 1
        # 1-cosin would be a measure that the least one would be 0 and highiest one 2
        
        
        
        # def index_max(a):
        #     return(np.unravel_index(np.argmax(a, axis=None), a.shape))
        
        print("--------estimate best number of cluster by applying elbow method ",)
        nx = normalise(X)
        xdot = 1 - nx.dot(nx.T)
        
        wcss = []  # the list of inertia for each number of cluster
        for i in range(2, max_cluster):
            
            kmedoids = KMedoids(n_clusters=i, random_state=0).fit(xdot)
        #    print(kmedoids.inertia_, "for kmedoids with n_cluster=",i)
            wcss.append(kmedoids.inertia_)
        
        wcss = np.array(wcss)
        # find the optimal # of cluster
        n_cluster = optimal_number_of_clusters(wcss)
        print("Best number of cluste would be:",n_cluster)
        # using kmedoids technic to find the clustring lables for sentenses
        kmedoids = KMedoids(n_clusters=n_cluster, random_state=0).fit(xdot)
        # creat a dataframe from setences and their cluster label
        clusterd_df_kmedoids = pd.DataFrame({'sentence': sentsp_pure})
        clusterd_df_kmedoids['class'] = kmedoids.labels_
        
        # creat a datarame from cluster labels , number of sentences in each label, the medoids sentence
        # and the index of medoids sentence in the orginal document
        unique, counts = np.unique(kmedoids.labels_, return_counts=True)
        
        clusterd_df_stats = pd.DataFrame({'label': unique})
        clusterd_df_stats['count'] = counts
        clusterd_df_stats['medoids_sentence'] = np.array(sentsp_pure)[kmedoids.medoid_indices_]
        clusterd_df_stats['mediods_sentence_indice'] = kmedoids.medoid_indices_
        return(clusterd_df_kmedoids,clusterd_df_stats)
    
    from statistics import median,mean

    #sample clustring the first input file
    print(proj_df.head())
    corpus_str = proj_df.org_doc[0]
    cleaned_str = proj_df.doc[0]
    sents_start = proj_df['sentence_start'][0]
    clusterd_sents,clustered_medoids=cluster_sentences(corpus_str,cleaned_str,sents_start)

    #estiamte a measure of similarity between each paragraph sentences and benchmark

    sim_to_bench=[]
    doc_sim_threshold=0.5
    A = normalise(w2vec_model.wv[bench])
    for clust in clusterd_sents['class'].unique():
        sub_clust= clusterd_sents[clusterd_sents['class']==clust]
        sub_doc="".join(list(sub_clust.sentence.values))
    #   sub_doc  =list(set(sub_doc.split()).intersection(set(vocab)))
    #exteract words wtha have entry in vocab only 
        sub_doc  =[item for item in sub_doc.split() if item in vocab]
        #normalize the document words vectors
        B_words = normalise(w2vec_model.wv[sub_doc])
        cosin_matrix=A.dot(B_words.T) # cosine matrix between doc words and benchmarks
    # find maximum cosin of all benchmark for each doc word    
        w_max_pool=cosin_matrix.max(axis=1)
        #weight of the benchmark with maximum similarity to sentences
        w_max_pool_weight= bench_w[cosin_matrix.argmax(axis=1)%cosin_matrix.shape[0]]
    #aggrigate words with similarity more than doc_sim_thershold weights     
        doc_total_similarity =np.sum((w_max_pool>doc_sim_threshold)
                                    *w_max_pool_weight)
    #    doc_total_similarity =median(w_max_pool)
    #    doc_total_similarity =mean(w_max_pool)
        sim_to_bench.append(doc_total_similarity)  
    norm_sim =sim_to_bench/np.sum(sim_to_bench)
    #%% highlight the sentnces witg highier similarity to benchmark

    def hlight(page,search_term):
        
        matching_val_area = page.search_for(search_term,quads=True)
        if(matching_val_area):
        #                    print(matching_val_area)
            highlight = page.add_highlight_annot(matching_val_area)
            highlight.update()  
        else:
            for sub_search_term in search_term.split('\n'):
                if(len(sub_search_term)!=0):
                    matching_val_area = page.search_for(sub_search_term,quads=True)
                    if(len(matching_val_area)!=0):
            #                    print(matching_val_area)
                        highlight = page.add_highlight_annot(matching_val_area)
                        highlight.update() 
                    else:                        
                        print("NO SUBMATCH FOUND:<<",sub_search_term,">>")
        
    def highlight_sents(proj_dir,doc_ind,proj_df,all_sents):
    #   fpath = os.path.join(proj_dir, "++%s"%(doc_ind))
        
    #    pdfdoc = fitz.open(fpath)
        output_buffer = BytesIO()
        path = pathlib.Path('Highlighted_files')
        path.mkdir(parents=True, exist_ok=True)
        output_file="static/Highlighted_files/HighLighted++%s"%(doc_ind)
        print("creat highlited outpout file:",output_file)

        sents_start = proj_df['sentence_start'][doc_ind]
        
        page_start=np.array(proj_df['page_start_list'][doc_ind])
        pdfdoc=proj_df['fitz_doc'][doc_ind]
        
        for ind_sent in range(len(all_sents)):
            if(all_sents[ind_sent]):
                start = sents_start[ind_sent]
                end   = sents_start[ind_sent+1]
                page_ind_start=np.sum(page_start<=start)-1
                page_ind_end=np.sum(page_start<=end)-1            
                if(page_ind_start==page_ind_end):                
                    page  = pdfdoc[int(page_ind_start)]
                    search_term =proj_df['org_doc'][doc_ind][start:end]
                    hlight(page,search_term)
                else:
                    page  = pdfdoc[int(page_ind_start)]
                    search_term =proj_df['org_doc'][doc_ind][start:page_start[page_ind_start+1]]
                    hlight(page,search_term)
                    page  = pdfdoc[int(page_ind_start+1)]
                    search_term =proj_df['org_doc'][doc_ind][page_start[page_ind_start+1]:end]
                    hlight(page,search_term) 
    #                print("two pages sentence",page_ind_start)
        
        pdfdoc.save(output_buffer)
    #   pdfdoc.close()       

        with open(output_file, mode='wb') as f:
            f.write(output_buffer.getbuffer() )             
        print("-------------------------------------------------------------")        

    # %% esitmating a measure for showing how a sentnces could be related 
    # to the benchmarks. the weighted average vector of each snetnce is used as 
    # a represntator of each sentence


    # COSINE CALCULATION
    # normalize the benchmark words vector
    A = normalise(w2vec_model.wv[bench])


    # a similarity datarame has been build to save the similarity of each bench mark 

    # the documents index (rows) are benchmarks and columns would be input project
    """
    This way strongly suggests we do an additional time save, though. 
    The formula for the cosine similarity of two vectors is u.v / |u||v|, 
    and it is the cosine of the angle between the two. Because we're iterating, 
    we keep recalculating the lengths of the rows of B each time and throwing the
    result away. A nice way around this is to use the fact that cosine similarity 
    does not vary if you scale the vectors (the angle is the same). 
    So we can calculate all the row lengths only once and divide by them to make 
    the rows unit vectors. And then we calculate the cosine similarity simply 
    as u.v, which can be done for arrays via matrix multiplication. 
    """
    # iterate over all input documents calculate cosine similarity of each words in doc
    # and each benchmarks .
    # for each benchmarks find Probability Distribution with 2 bins over all words of documnets
    doc_sim_threshhold=.5
    sim_to_bench=[]
    for doc_ind, proj in proj_df.iterrows():
        print("-------processing project:", doc_ind)
        corpus_str = proj_df.org_doc[doc_ind]
        cleaned_str = proj_df.doc[doc_ind]
        sents_start = proj_df['sentence_start'][doc_ind]
        page_start=np.array(proj_df['page_start_list'][doc_ind])
        pdfdoc=proj_df['sentence_start'][doc_ind]
        # select the words from documnets which have entery in vocab
        sub_doc  =list(set(cleaned_str.split()).intersection(set(vocab)))
        #normalize the document words vectors
        B_words = normalise(w2vec_model.wv[sub_doc])
        cosin_matrix=A.dot(B_words.T) # cosine matrix between doc words and benchmarks
    # maximum cosin of all benchmark for each doc word weighted    
        w_max_pool=cosin_matrix.max(axis=1)*bench_w[cosin_matrix.argmax(axis=1)%cosin_matrix.shape[0]]
        doc_total_similarity =np.sum(w_max_pool>doc_sim_threshhold)/len(w_max_pool)
        sim_to_bench.append(doc_total_similarity)
        
        
        sentsp_pure = []  # list of sentences that have at least one word in vocab
        sents_list = []   # list of index of the sentences in orginal sentences list
        sents_weighted_vector = []  # list of the  weighted maximum similarity to benchmark
        sents_benchw = []
        sents_bench_similarity = []
        for ind in range(len(sents_start)-1):  # loop over the sentenses
            start = sents_start[ind]
            end = sents_start[ind+1]
            page_ind_start=np.sum(page_start<start)-1
            page_ind_end=np.sum(page_start<end)-1
            # split words cleaned versio of sentence
            focus_sents = cleaned_str[start:end].split(' ')

            # select the words from semtence which have entery in vocab
            sub_sent = list(set(focus_sents).intersection(set(vocab)))
            # sentence words vectors  multiply by the content measure
            # which present the rarreness and calculate the average of the weighted sentences vectors
            if(len(sub_sent) != 0):
                # normalized sentonce vectors
                B = normalise(w2vec_model.wv[sub_sent])
    # calculate dot production of the sentence words and benchmarks  vectors to
    # creat a similarity matrix between all sentens words and all benchmarks
    # calculate weighted average of max similarity of each word in sentence to entire benchmark
    # and applying the weight associated to the benchmark with maximum simialrity to the word
                ppp = A.dot(B.T)
    #1. for each word in sentense find the max cosine with all benchmarks
    #2. this max value multipy by its corresponded weight in benchmark
    #3. these multiplication sumup and divided by the sum of applied weights to 
    #calculate the weighted average of similarity to the becnhmarks for words in a sentence
                avg = np.sum(ppp.max(
                    axis=0)*bench_w[ppp.argmax(axis=0)])/len(sub_sent)
                # avg = np.sum(ppp.max(
                #     axis=0)*bench_w[ppp.argmax(axis=0)])/(np.sum(bench_w[ppp.argmax(axis=0)]))
    #            avg=np.sum(bench_w[ppp.argmax(axis=0)][ppp.max(  axis=0)>similarity_tershold])
            else:  # if there is not any word in sentense add 0
                avg=0
            sents_benchw.append(avg)
                
            # original sentence in benchmark
            sentsp_pure.append(corpus_str[start:end])
        
        sents_benchw = np.array(sents_benchw)
        sents_df = pd.DataFrame({'setence': sentsp_pure})
        sents_df['similarity_to_bench'] = sents_benchw
        all_sents =sents_df['similarity_to_bench']>similarity_tershold
    #highlight the sentences which have similarity to benchmarks more than thershold    
        highlight_sents(proj_dir,doc_ind,proj_df,all_sents) 

    #put the similarity to becnh for each document     
    proj_df['simimlarity_to_bench']=sim_to_bench


    proj_df['simimlarity_to_bench'].to_csv(sim_to_bench_file)
    print('SImilarity to the benchmark for each input file is written to :',sim_to_bench_file)
        # creat an numpy array from the weighted centroid vector of all sentences
    print("-------------------------------------")
    print("----End of the process")

    return(True)