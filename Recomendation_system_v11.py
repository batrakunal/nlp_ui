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
import collections

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
sns.set_theme()

from flask import session

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
    def find_dominanat_font(page):
        """
        Find the most pouplar (fint,size) in entire document 
        this could be best indicator of the most part of document and 
        the footer and noter usually have diffrence font,size attribute

        Parameters
        ----------
        doc : pyMUpdf documnet format
            
        examine an input pymupdf file and find the most applcable font in file
        it collect all bloacks applied fonts and for all of them find frequency of use. the font and size are 
        used together to find the most applied one.

        Returns
        -------
        the most applied tupple ('font_name',size).

        """
        pg_font=[]
        #itterate ove the document pages
        # for k in range( len(list(doc))):
        #     page =list(doc)[k]
            # access to the dict of the page
            #https://pymupdf.readthedocs.io/en/latest/textpage.html#TextPage.extractDICT
        dic =page.get_text("dict", sort=True)
        blks_font=[]
        blks_size=[]
        blks_no=[]
        #itterate over the block of the page
        for blks in dic['blocks']:
            blk_n=blks['number']
            #examine if the blks is a text only block
            if(blks['type']==0):
                lns_font=[]
                lns_size=[]     
                #itterate over lines of the block
                for lns in blks['lines']:  
                    #iterrate over spans of each line
                    for spns in lns['spans']:
                        #collect font type and size of each span
                        lns_size.append(int(spns['size']))
                        lns_font.append((spns['font'],int(spns['size'])))
                blks_font.append(lns_font)
                blks_size.append(lns_size)
                blks_no.append(blk_n)
        #find frequency of font,size applied in each page
        frequency = collections.Counter([x for y in blks_font for x in y]) 
    #        print(frequency.most_common(1))
        pg_font.append(frequency.most_common(1))
        #find the most used font,size and return it as result
        dominanat_font = collections.Counter([x[0] for y in pg_font for x in y]).most_common(1)[0][0]
        return(dominanat_font)


    def page_text(page):
        """
        find the blocks of text in page with domminant_font and concatinate them 
        to return the page content

        Parameters
        ----------
        page : page of document in pymupdf format
            DESCRIPTION.
        dominanat_font : dominate font in document as ('font_name',size)
            DESCRIPTION.

        Returns
        -------
        blk: the raw text in documnet
        start_ln: the starting pointer list of all lines in the returned text
        bbox_list: the quad coordination of bbox of each line in document .

        """
        #get dict for the document it contins many information about the page 
        #content
        #https://pymupdf.readthedocs.io/en/latest/textpage.html#TextPage.extractDICT
        #https://pymupdf.readthedocs.io/en/latest/page.html?highlight=get_text#Page.get_text
        dic =page.get_text("dict", sort=True)
        blks_font=[]
        blks_no=[]
        blks_text=[]
        blks_bbox=[]
        #itterate over the blocks in document
        for blks in dic['blocks']:
            blk_n=blks['number']
            #if blks contains text and not graph
            if(blks['type']==0):
                lns_font=[]
                lns_text=[]
                lns_bbox=[]
                #itterate over lines in each block
                for lns in blks['lines']:  
                    spn_font=[]
                    spn_text=[]
                    spn_bbox=[]                #itterate over span of each line and collect fon and size and text in spane
                    for spns in lns['spans']:
                        spn_font.append((spns['font'],int(spns['size'])))
                        spn_text.append(spns['text'])
                        spn_bbox.append(spns['bbox'])
                    #record the lines bbox with its text
                    lns_bbox.append(spn_bbox)
                    lns_font.append(spn_font)
                    lns_text.append(spn_text)

                    
                blks_font.append(lns_font) #block fonts
                blks_no.append(blk_n) #blcok sequence number
                blks_bbox.append(lns_bbox) #block bbox
                blks_text.append(lns_text) #block text
            else: #for graph blocks only insert null fields
                blks_bbox=blks_bbox+[""]
                blks_font.append([(' ',0)])
                blks_no.append(blk_n)
                blks_text=blks_text+[""]           
        sel_blks_no=[]
        #itterate over collected blocks with its natual flow of block_no
        for kk in np.argsort(blks_no):
            # ind=blks_no[kk]
            # frequency=collections.Counter(blks_font[kk])
        #record only blocks that has the domminate font
    #it assumend the main body only use mostly the dominant font                     
    #        if(frequency.most_common(1)[0][0]==dominanat_font):
            sel_blks_no.append(kk)
    #        else:
    # #           print("ignore blk:%d"%(ind))
        sel_blks_no= np.array(sel_blks_no)

        blk=""
        start=0
        start_spn=[]
        bbox_list=[]
        #ittterate over slected blcoks and record the starting postion in returned
        #string for each line and its bbox
        #conect lines with \n
        for k in sel_blks_no:
            for kk in range(len(blks_text[k])):
                
                for kkk in range(len(blks_text[k][kk])):
                    start=len(blk)
                    start_spn.append(start)
                    bbox_list.append(blks_bbox[k][kk][kkk])
                    blk=blk+" "+blks_text[k][kk][kkk]
                blk=blk+"\n"

        return(blk,start_spn,bbox_list)

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


    def extract_text_from_folders_of_pdfs(proj_dir, stopwords1,
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
                columns=['file_name', 'org_doc', 'doc', 'page_start_list',
                        'sentence_start','start_ln','bbox_list','fitz_doc'])
        else:
            orgdoc_df = pd.DataFrame(columns=['file_name', 'doc'])

        i = 0

        for(root, dirs, files) in os.walk(proj_dir, topdown=True):
            for fname in files:
                if (fname.endswith('.pdf')): #only pdf will processed but it can process word and txt file as well
                    print("---processing<<%s>>"%(fname))
                    try:
                        fpath = os.path.join(root, fname)
                        doc = fitz.open(fpath)  # open document
                        rawText = ""  # open text output
                        page_start = []  # list of starting point of each page
    #                    sentnce_start_list
                        # set to ignore warning and error messge from mupdf
                        fitz.TOOLS.mupdf_display_errors(False)
    #                    dominanat_font =find_dominanat_font(doc)
                        page_start_ln=[]
                        page_bbox_list=[]
                        for page in doc:  # iterate the document pages
                            # get plain text (is in UTF-8)
                            text,start_ln,bbox_list=page_text(page)
    #                        text = page.get_text("text", sort=True)
                            if(text != ""):
                                # add the starting point of the new page to page list
                                page_start.append(len(rawText))
                                page_start_ln.append([x+len(rawText) for x in start_ln])
                                rawText = rawText+str(text)  # write text of page
                                page_bbox_list.append(bbox_list)                           
                        # check if the lenght of text is at least doc_min_len charachter
                        if(len(rawText) > doc_min_len):
                            txt, _ = ngramming_bforce2(rawText, stopwords1,
                                                    word_len=2, ngram_range=(2, 3),
                                                    bigram_base_frequency=.28,
                                                    trigram_base_frequency=.28)
                            if(OrgDoc):
                                sentences=find_sentences(rawText)
                                orgdoc_df.loc[len(orgdoc_df)] = [
                                    fname, rawText, txt, page_start, sentences,
                                page_start_ln,page_bbox_list, doc]
                            else:
                                orgdoc_df.loc[len(orgdoc_df)] = [
                                    fname, txt.split()]

                            print(i, '+<<', fname, ">> processed")
                            i += 1
                            if(i == max_files):
                                return orgdoc_df.set_index('file_name')
                        else:
                            print(
                                fname, "documnet is too short, not processed", len(rawText))
                        print("-------------------------------------\n\n\n")

                    except Exception as e:
                        print(e, '-', fname, ': pdf is not readable')
                        pass
        print("\n\n########### End of input document files preprocessing#########\n\n")
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

    # creat cluster for benchmarks
    def bench_clustring(bench_df):
        """
        accept the benchmarks dataframe and add a catagory as 'class' and also 
        medoids pharse for each bencmmarks catagory. 

        Parameters
        ----------
        bench_df : Dataframe from the benchmarks
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # normalize the benchmark words vector
        bnch=bench_df.benchmark
        A_bench = normalise(w2vec_model.wv[bnch])
        xdot = 1 - A_bench.dot(A_bench.T)
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
        
        bench_df['class'] = kmedoids.labels_

        # creat a datarame from cluster labels , number of sentences in each label, the medoids sentence
        # and the index of medoids sentence in the orginal document
        unique, counts = np.unique(kmedoids.labels_, return_counts=True)
        print("%d benchmarks are clustered into %d \nThe number of bench in each cluster is:"%(len(bnch),len(unique)),counts)

        bench_df['class_mediods_bench'] = [bnch[kmedoids.medoid_indices_[x]] for x in bench_df['class'].values]

    """
    -----main
    """

    """
    parapters used in the program
    """
    #################  INPUT FILES
    # Defining the path to the trainig pdf files
    folder_path = 'c:/dau'
    # folders for input files to be evaluated
    proj_dir = user_file_path
    # benchmarks file
    file_bench = 'Bench_weighted_rooted.csv'
    # stop words file
    stp_file = 'stopwords_en.txt'

    # names for saving and loading the trained model
    model_file = 'word2vec_v1.model'

    ################# OUTPUT FILES

    #csv file path
    table_folder = "tables"
    table_folder_path = os.path.join(user_file_path, table_folder)
    if not os.path.exists(table_folder_path):
        os.makedirs(table_folder_path)

    #Presenting the similarity degree for each file in proj_dir to the entire benchmark
    sim_to_bench_file = table_folder_path+"/"+'sim_to_bench.csv'
    #Presenting the similarity degree for each file in proj_dir to the entire benchmark
    sim_to_bench_detail_file = table_folder_path+"/"+'sim_to_bench_detail.csv'

    sim_to_bench_class_file = table_folder_path+"/"+'sim_to_bench_class.csv'
    ######### CONFIGURATION FACTORS
    # maximum number of fils that has been read for training
    max_num_files = 2000
    #minimum of accepteable word length
    word_min_len = 2
    ngram_min = 2  # minimum ngraming
    ngram_max = 3  # maximum ngraming

    # maximum # of cluster to estimate the data clustring wcss
    max_cluster = 50


    # the text length would be exponentiation to (-base_frequency) to calculate minimum
    # frequency of the ngrams to be considered.
    # the higher the number, the less bigrams will be extracted
    bigram_base_frequency = 0.28
    # the higher the number, the less trigrams will be extracted
    trigram_base_frequency = 0.28

    # this treshhold is very important factor that select the sentences with highier 
    #weighted value compare to the tershold. total passed thershold words are divided by total 
    #number of doc words to calculate level of relation to the benchmark
    # highier value limit the number of sentence and vice versa
    # this value has been completly direct relation with the value which is used to smooth the average similarity
    #to benchmarks. I use (number of words)** .7 as divider to make an average in othe word
    #the average of the similarity of the words to benchmarks are multiply to (number of words)**.3
    # this factor smootly bring the sentence length ino account. it means longer sentence has more value than shorter
    snetence_len_factor =0.3

    # the treshhold which is used to select related sentences to benchmark
    similarity_tershold = 1.0 #

    # a treshold to find the words which has at least that amount of similarity to benchmarks
    doc_sim_threshold=.5
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
    #************very  processing expensive took hours & memmory consuming
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
    # rarness factor present as CONTENT measure
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

    print("from total {:,} words(ngrams) {:,} unique words(ngrams) are added to vocabulary".format(
        tot_cnt, len(vocab)))

    # calculate content measure from the word frequency in vocabulary inspired by:
    # (Duong et al. 2019) it present the rareness of a word which make it more valuable
    # in a sentence to be concentrated as a key word
    #
    voc_cnt['content'] = [-np.log(x/tot_cnt) for x in voc_cnt['count']]
    voc_cnt['content'] /= voc_cnt['content'].max()
    # set the word as key for accesing vocabulary contents
    voc_cnt.set_index('word', inplace=True)



    # caculate number of benchmarks that are not preseted in vocabulary
    bench_dif = list(set(bench)-set(vocab))
    bench_found = list(set(bench).intersection(set(vocab)))
    print("\n\n----Number of unknown words from bench in vocabulary:", len(bench_dif),
        "from:", len(set(bench)))
    print("------unknown words from bench which are ignored:\n", bench_dif)
    print("----------------------------------------------------")
    # ignore benchmarks wich is not in vocabulary
    bench_w = [bench_w[i] for i in range(len(bench)) if bench[i] in bench_found]
    bench = [bench[i] for i in range(len(bench)) if bench[i] in bench_found]
    # normalized bench marks weight between minimum and 1.
    bench_w /= np.max(bench_w)
    # pd.DataFrame(bench_dif).to_csv('Benchmarks_missing.csv')
    # creat a df from the benchmarks and their weights
    bench_df = pd.DataFrame({'benchmark': bench, 'weight': bench_w})
    bench_df = pd.merge(left=bench_df, right=voc_cnt, left_on='benchmark', right_on='word')[
        ['benchmark', 'weight', 'count', 'content']]
    #add catagory to benchmakr dataframe
    bench_clustring(bench_df)
    bench_df=bench_df.sort_values(by=["class"]).reset_index(drop=True)
    bench=bench_df['benchmark'].values
    bench_w=bench_df['weight'].values
    # create vocabulry df from vector, len,and count
    # voc_df = voc_cnt.copy()
    # voc_df[["d%03d" % (i) for i in range(300)]] = w2vec.wv[vocab]
    # # creat a sample vocabulary by count>50 and 20<len<25

    # sample_vocab = voc_df[(voc_df['count'] > 50) & (
    #     voc_df['len'] > 20) & (voc_df['len'] < 25)]
    #read the input documents int a dataframe
    proj_df = extract_text_from_folders_of_pdfs(proj_dir,
                                                stopwords1, bench=bench, OrgDoc=True)#,max_files=2)

    # %%

    # this part load a corpus and its sentences boundaries 
    # it use that to calculate the weighted average of each sentences vector
    # for weight the content factor of each word in vocabulary is used
    # this means a rare word is more important than a popular word for calcualting
    # the average. the weighted average would be feed to a kmediods model and find the
    # optimal number of cluster by applying elbow technic. the sentences in corpus will
    # clsuterd by the best number of cluster to find the related sentences in curpus.



    def cluster_sentences_wmd(corpus_str,cleaned_str,sents_start):
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
        sents_cleaned_list = []
        for ind in range(len(sents_start)-1):
            start = sents_start[ind]
            end = sents_start[ind+1]
            focus_sents = cleaned_str[start:end].split(' ')
        
            # select the words from sentence which have entery in vocab
            sub_sent = list(set(focus_sents).intersection(set(vocab)))
            # document words vectors  multiply by the content measure
            # which present the rareness and calculate the average of the weighted sentences vectors
            if(len(sub_sent) != 0):
                sents_cleaned_list.append(sub_sent)
                sents_list.append(ind)
                sentsp_pure.append(cleaned_str[start:end])
        
        # creat an numpy array from the weighted centroid vector of all sentences
        wmd_matrix=np.zeros(shape=(len(sents_cleaned_list),len(sents_cleaned_list)))
        for i in range(len(sents_cleaned_list)):
            for j in range(len(sents_cleaned_list)):
                if(i==j):
                    wmd_matrix[i,j]=0
                elif(i>j):
                    distance = w2vec.wv.wmdistance(sents_cleaned_list[i], sents_cleaned_list[j])
                    wmd_matrix[i,j]=distance
                    wmd_matrix[j,i]=distance
        

        xdot = wmd_matrix
        
        
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

    # the list of the sum of squared distance between each point 
    #and the centroid in a cluster for each number of cluster    
        wcss = []  
        for i in range(2, max_cluster):
            
            kmedoids = KMedoids(n_clusters=i, random_state=0).fit(xdot)
        #    print(kmedoids.inertia_, "for kmedoids with n_cluster=",i)
            wcss.append(kmedoids.inertia_)
        
        wcss = np.array(wcss)
        # find the optimal # of cluster
        n_cluster = optimal_number_of_clusters(wcss)
        print("Best number of cluster would be:",n_cluster)
        # using kmedoids technic to find the clustring lables for sentenses
        kmedoids = KMedoids(n_clusters=n_cluster, random_state=0).fit(xdot)
        # creat a dataframe from setences and their cluster label
        clusterd_df_kmedoids = pd.DataFrame({'sentence': sentsp_pure})
        clusterd_df_kmedoids['class'] = kmedoids.labels_
        all_sents_clusters=np.array([-1]*(len(sents_start)-1))
        all_sents_clusters[sents_list]=kmedoids.labels_
        # creat a datarame from cluster labels , number of sentences in each label, the medoids sentence
        # and the index of medoids sentence in the orginal document
        unique, counts = np.unique(kmedoids.labels_, return_counts=True)
        
        clusterd_df_stats = pd.DataFrame({'label': unique})
        clusterd_df_stats['count'] = counts
        clusterd_df_stats['medoids_sentence'] = np.array(
            sentsp_pure)[kmedoids.medoid_indices_]
        clusterd_df_stats['mediods_sentence_indice'] = kmedoids.medoid_indices_
        return(all_sents_clusters,clusterd_df_kmedoids,clusterd_df_stats)


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
        
            # select the words from sentence which have entery in vocab
            sub_sent = list(set(focus_sents).intersection(set(vocab)))
            # document words vectors  multiply by the content measure
            # which present the rareness and calculate the average of the weighted sentences vectors
    #########  I decided to not use CONTENT factor!!!
            if(len(sub_sent) != 0):
                # w_vec = np.sum(w2vec_model.wv[sub_sent].T * voc_cnt.loc[sub_sent].content.values,
                #                axis=1) / voc_cnt.loc[sub_sent].content.sum()
                w_vec = np.average(w2vec_model.wv[sub_sent],axis=0)
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
        
    #    print("--------estimate best number of cluster by applying elbow method ")
    #normalize the sentence vectors
        nx = normalise(X)
    #find the cosine similarity between the e=sentences by dot product and
    #change the measure to a distance 
        xdot = 1 - nx.dot(nx.T)
    # wcss is the list of the sum of squared distance between each point 
    #and the centroid in a cluster for each number of cluster      
        wcss = []  
        for i in range(2, max_cluster):
            
            kmedoids = KMedoids(n_clusters=i, random_state=0).fit(xdot)
        #    print(kmedoids.inertia_, "for kmedoids with n_cluster=",i)
            wcss.append(kmedoids.inertia_)
        
        wcss = np.array(wcss)
        # find the optimal # of cluster
        n_cluster = optimal_number_of_clusters(wcss)
    #    print("Best number of cluste would be:",n_cluster)
        # using kmedoids technic to find the clustring lables for sentenses
        kmedoids = KMedoids(n_clusters=n_cluster, random_state=0).fit(xdot)
        # creat a dataframe from setences and their cluster label
        clusterd_df_kmedoids = pd.DataFrame({'sentence': sentsp_pure})
        clusterd_df_kmedoids['class'] = kmedoids.labels_
        all_sents_clusters=np.array([-1]*(len(sents_start)-1))
        all_sents_clusters[sents_list]=kmedoids.labels_
        # creat a datarame from cluster labels , number of sentences in each label, the medoids sentence
        # and the index of medoids sentence in the orginal document
        unique, counts = np.unique(kmedoids.labels_, return_counts=True)
        
        clusterd_df_stats = pd.DataFrame({'label': unique})
        clusterd_df_stats['count'] = counts
        clusterd_df_stats['medoids_sentence'] = np.array(
            sentsp_pure)[kmedoids.medoid_indices_]
        clusterd_df_stats['mediods_sentence_indice'] = kmedoids.medoid_indices_
        return(all_sents_clusters,clusterd_df_kmedoids,clusterd_df_stats)

    #%% highlight the sentnces with highier similarity to benchmark

    def hlight(page,search_term,start):
        """
        Parameters
        ----------
        page : TYPE
            pyMuPDF page object  .
        search_term : string
            an string which want to highlited in page.
        start : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """    
        if(len(search_term)>1):
            matching_val_area = page.search_for(search_term,quads=True)
            if(len(matching_val_area)!=0):
            #                    print(matching_val_area)
                highlight = page.add_highlight_annot(matching_val_area)
                highlight.update()  
        
            else:
                for sub_search_term in search_term.split('\n'):
        
                    if(len(sub_search_term)>5): # at least two charachter would be there
                        matching_val_area = page.search_for(sub_search_term,quads=True)
                        if(len(matching_val_area)!=0):
                            highlight = page.add_highlight_annot(matching_val_area)
                            highlight.update() 
                        else: #one more time try on finding the subterm by ignoring last character
                            matching_val_area = page.search_for(sub_search_term[:-1],quads=True)
                            if(len(matching_val_area)!=0):
                                highlight = page.add_highlight_annot(matching_val_area)
                                highlight.update() 
                            else:    
                                print("NO SUBMATCH FOUND:<<",sub_search_term,">>",start)



    def highlight_sents(proj_dir,doc_ind,proj_df,all_sents):
        """
        in ans specifc input file in proj_df with doc_ind all_sents selected
        are highlited in pyMuPDF object and the result wrote to a outputfile 
        in a director "Highlighted_files/HighLighted++%s" which %s is the same input
        file name

        Parameters
        ----------
        proj_dir : DataFrame
            input project directory name  .
        doc_ind : TYPE
            doc index in proj_df.
        proj_df : TYPE
            Dataframe contains input document files.
        all_sents : TYPE
            sentence which must be highlited.

        Returns
        -------
        None.

        """
    #   fpath = os.path.join(proj_dir, "++%s"%(doc_ind))
        
    #    pdfdoc = fitz.open(fpath)
        output_buffer = BytesIO()
        path = pathlib.Path('Highlighted_files')
        path.mkdir(parents=True, exist_ok=True)
        output_file="static/user_files/"+session['userfolder']+"/"+session['curr_file'][0]+"/HighLighted++%s"%(doc_ind)
    #    print("create highlited outpout file:",output_file)

        sents_start = proj_df['sentence_start'][doc_ind]
        page_start=np.array(proj_df['page_start_list'][doc_ind])
        pdfdoc=proj_df['fitz_doc'][doc_ind]
        for ind_sent in range(len(all_sents)):
            if(all_sents[ind_sent]):
                start = sents_start[ind_sent]
                end   = sents_start[ind_sent+1]

                strt_ln=[x for y in proj_df.loc[doc_ind,'start_ln'] for x in y]
                bbox_l= [x for y in proj_df.loc[doc_ind,'bbox_list'] for x in y]
                for k1 in range(len(strt_ln)-1):  
                    s=proj_df.loc[doc_ind,'org_doc']
                    s=s[strt_ln[k1]:strt_ln[k1+1]]


                    if((strt_ln[k1]>=start) & (strt_ln[k1+1]<=end)) |\
                    ((strt_ln[k1]<=start) & (strt_ln[k1+1]>=start) )|\
                    ((strt_ln[k1]<end) & (strt_ln[k1+1]>end)):
                            page_ind=np.sum(page_start<=strt_ln[k1])-1
                            page  = pdfdoc[int(page_ind)]
                            highlight = page.add_highlight_annot(bbox_l[k1])
                            if(highlight!=highlight):
                                print("no highlight found!",page_ind,
                                    proj_df['org_doc'][doc_ind][strt_ln[k1]:strt_ln[k1+1]])

        pdfdoc.save(output_buffer)
    #   pdfdoc.close()       

        with open(output_file, mode='wb') as f:
            f.write(output_buffer.getbuffer() )             
    #    print("-------------------------------------------------------------")        

    # %% esitmating a measure for showing how a sentnces could be related 
    # to the benchmarks. the weighted average vector of each snetnce is used as 
    # a represntator of each sentence



    ####################################
    def distribution_cal(cosin_matrix):
        """
    #make a 2 bins histogram to implement the probability distribution of 
    # the words similarity to each benchmarks
    # as the cosine parameter has a variance between -1,1 the 2 bin probability
    #distribution has 2 bin (-1,0) and (0,1) the similarity of two vector would be
    # the bin (0,1) as second bin. this will be show the similarity
    #this part extract as similarity of the doc to specific benchmark
    #    

        Parameters
        ----------
        cosin_matrix : TYPE
            A matrix of cosine similarity between two comparing set of vectors
            it returns distribution of cosine more than 0 for each row.

        Returns
        -------
        it returns distribution of cosine more than 0 for each row as a numpy
        array.

        """
        bins=2
        hist= np.zeros((cosin_matrix.shape[0],bins))
        for k in range(hist.shape[0]):
            hist[k,:],rng=np.histogram(
            cosin_matrix[k,:],
            bins=bins,
            range=(-1, 1),
            density=True) 
        return(hist[:,1])    
    #calculate the distance between all snetnces and benchmark with WMD
    def wmd_sentence(bench,clusterd_sents):
        sent_dist=[]
        for sent in clusterd_sents['sentence']:
                # select the words from semtence which have entery in vocab
                sub_cluster = list(set(sent.split()).intersection(set(vocab)))        
                if(len(sub_cluster) != 0):
                    # normalized sentonce vectors
                    distance =  w2vec_model.wv.wmdistance(sub_cluster, bench)
                    sent_dist.append(distance )
                else: sent_dist.append(-1)
        clusterd_sents['WMD']=sent_dist


    # calculate dot production of the sentence words and b
    
    # COSINE CALCULATION
    # normalize the benchmark words vector
    A_bench = normalise(w2vec_model.wv[bench])


    # a similarity dataframe has been build to save the similarity of each bench mark 

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

    sim_to_bench=[]
    sim_to_bench_sent=[]
    proj_df['cluster_list']=[[] for i in range(len(proj_df))] #list of cluters in each doc
    proj_df['number_of_cluster']=0 #number of cluster for each input doc

    #proj_df['sentences_vect']=[[]*len(proj_df)]
    sim_df=pd.DataFrame(data=0,index=bench,columns=proj_df.index)
    #%%
    print("\n\n-------------------\nProcessing input files:\n\n")
    for doc_ind, proj in proj_df.iterrows():
        print("--Processing Document:  ", doc_ind)
        corpus_str = proj_df.org_doc[doc_ind]
        cleaned_str = proj_df.doc[doc_ind]
        sents_start = proj_df['sentence_start'][doc_ind]
        page_start=np.array(proj_df['page_start_list'][doc_ind])
        pdfdoc=proj_df['sentence_start'][doc_ind]
        # select the words from documnets which have entery in vocab
        sub_doc  =list(set(cleaned_str.split()).intersection(set(vocab)))
        #normalize the document words vectors
        B_words = normalise(w2vec_model.wv[sub_doc])
        cosin_matrix=A_bench.dot(B_words.T) # cosine matrix between doc words and benchmarks
        
        
        #make a 2 bins histogram to implement the probability distribution of 
        # the words similarity to each benchmarks
        # as the cosine parameter has a variance between -1,1 the 2 bin probability
        #distribution has 2 bin (-1,0) and (0,1) the similarity of two vector would be
        # the bin (0,1) as second bin. this will be show the similarity
        #this part extract as similarity of the doc to specific benchmark
        #
    ####################################
    
        sim_df[doc_ind]=distribution_cal(cosin_matrix)

        print("--calculating best # of Document sentences' clusts applying Elbow method")
        all_sents_clusters,clusterd_sents,clustered_medoids=cluster_sentences(corpus_str,cleaned_str,sents_start)
    #    all_sents_clusters_wmd,clusterd_sents_wmd,clustered_medoids_wmd=cluster_sentences_wmd(corpus_str,cleaned_str,sents_start)
        print("--Best number of cluster is:",len(np.unique(clusterd_sents['class']))-1)
        proj_df.at[doc_ind,'number_of_cluster']=len(np.unique(clusterd_sents['class']))

    # this part estimate each cluste(smart paragraph) similarity to entire benchmarks 
        bench_cat= len(np.unique(bench_df['class'])) # number of becnh catagory
        #entire bench similarity to clustered sentences
        sim_cluster=np.zeros(shape=(len(np.unique(clusterd_sents['class']))-1)) 
        #each benchmark cluster similarity to clustered sentences
        sim_cluster_benchs=np.zeros(shape=(len(np.unique(clusterd_sents['class']))-1,bench_cat))
        for ind in range(len(np.unique(clusterd_sents['class']))-1):
            cluster_corpus=" ".join(clusterd_sents[clusterd_sents['class']==ind]['sentence']).split()
            # select the words from sentence which have entery in vocab
            sub_cluster = list(set(cluster_corpus).intersection(set(vocab)))        
            if(len(sub_cluster) != 0):
                # normalized sentences vectors in each cluster
                B = normalise(w2vec_model.wv[sub_cluster])
    # calculate dot production of the sentence words and benchmarks  vectors to
    # creat a similarity matrix between all sentens words and all benchmarks
    # calculate weighted average of max similarity of each word in sentence to entire benchmark
    # and applying the weight associated to the benchmark with maximum simialrity to the word
                ppp = A_bench.dot(B.T)
    #1. for each word in sentense find the max cosine with all benchmarks
    #2. this max value multipy by its corresponded weight in benchmark
    #3. these multiplication sumup and divided by the sum of applied weights to 
    #calculate the weighted average of similarity to the becnhmarks for words in a sentence           
                sim_cluster[ind]=  np.sum(bench_w[distribution_cal(ppp) > doc_sim_threshold])/np.sum(bench_w)
    ####################################################################################################
    #this part create a matrix of similarity between each benchmark cluster and each sentences clusters           
                for i in range(bench_cat):
                    sub_bench= bench_df[bench_df['class']==i]['benchmark'].values
                    sub_bench_w =bench_df[bench_df['class']==i]['weight'].values
                    A_sub_bench=normalise(w2vec_model.wv[sub_bench])
                    ppp_sub=A_sub_bench.dot(B.T)
                    sim_cluster_benchs[ind,i] = np.sum(sub_bench_w[distribution_cal(ppp_sub) > doc_sim_threshold])/np.sum(sub_bench_w)
        sentsp_pure = []  # list of sentences that have at least one word in vocab
        sents_list = []   # list of index of the sentences in orginal sentences list
        sents_weighted_vector = []  # list of the  weighted maximum similarity to benchmark
        sents_benchw = []
        sents_bench_similarity = []
        sent_page=[]
        for ind in range(len(sents_start)-1):  # loop over the sentenses
            start = sents_start[ind]
            end = sents_start[ind+1]
            page_ind_start=np.sum(page_start<start)-1
            page_ind_end=np.sum(page_start<end)-1
            # split words cleaned versio of sentence
            focus_sents = cleaned_str[start:end].split(' ')

            # select the words from sentence which have entery in vocab
            sub_sent = list(set(focus_sents).intersection(set(vocab)))
            # sentence words vectors  multiply by the content measure
            # which present the rarreness and calculate the average of the weighted sentences vectors
            if(len(sub_sent) != 0):
                # normalized sentence vectors
                B = normalise(w2vec_model.wv[sub_sent])
    # calculate dot production of the sentence words and benchmarks  vectors to
    # creat a similarity matrix between all sentens words and all benchmarks
    # calculate weighted average of max similarity of each word in sentence to entire benchmark
    # and applying the weight associated to the benchmark with maximum simialrity to the word
                ppp = A_bench.dot(B.T)
    #1. for each word in sentense find the max cosine with all benchmarks
    #2. this max value multipy by its corresponded weight in benchmark
    #3. these multiplication sumup and divided by the sum of applied weights to 
    #calculate the weighted average of similarity to the becnhmarks for words in a sentence
    ########################################################           
                max_pool =ppp.max(axis=0)
                max_pool_w=bench_w[ppp.argmax(axis=0)]
                #avg=np.sum(max_pool*max_pool_w)/len(sub_sent)


                #to consider the impact of the sentence with large number of words
                # the weighted average only considerd average value that dose not
                #shows the value of sentence length. to bring to the picture the sentence len
                # the (number of word in sentence)**.3(snetence_len_factor) is used to 
                #consider this factor
                #with this change the thershold must be changed to around 1.0
                avg=np.sum(max_pool*max_pool_w)/np.sum(max_pool_w)
                #the average similarity multipy to a smooth factor of sentence len
                avg=avg* (np.sum(max_pool_w)**snetence_len_factor)
            else:  # if there is not any word in sentense add 0
                avg=0
            sents_benchw.append(avg)
            # original sentence in document
            sentsp_pure.append(corpus_str[start:end])
            # add page number for each sentence
            sent_page.append(np.sum(proj.sentence_start[ind]>=np.array(proj.page_start_list)))
    
        sents_benchw = np.array(sents_benchw)
    #    sents_benchw /=sents_benchw.max()
        sents_df = pd.DataFrame({'sentence': sentsp_pure})
        sents_df['similarity_to_bench'] = sents_benchw
        sents_df['page']=sent_page
        sents_df['cluster']=all_sents_clusters
        #find the snetences with similarity more than thershold
        all_sents =sents_df['similarity_to_bench']>=similarity_tershold
        sents_df_selection=sents_df[sents_df['similarity_to_bench']>similarity_tershold]
        sim_count=sents_df_selection['cluster'].value_counts()
    
        sim_count=sim_count/np.sum(sim_count)

    #highlight the sentences which have similarity to benchmarks more than thershold    
        highlight_sents(proj_dir,doc_ind,proj_df,all_sents) 
        print("--pdf file  with highlited recomendation parts are created:")
    #creat/use path 'clustring' to creat result csv file for each doc
        path = pathlib.Path('clustering')
        path.mkdir(parents=True, exist_ok=True)
    #write to a csv file clustering fro each sentence 
        output_file="clustering/clustered++%s"%(doc_ind.replace('.pdf', '.csv'))
        sents_df.to_csv(output_file,index_label="Seq.")
        print('--Document sentences clusters and similarity measures CSV file created')
    # write to csv file the each cluster similarity to entire becnhmark
        output_file="clustering/cluster_ranks++%s"%(doc_ind.replace('.pdf', '.csv'))
        cluster_ranks=pd.DataFrame(data={'cluster_number':sim_count.index,'similarity_to_bench':sim_count.values})
        cluster_ranks.sort_values(by='similarity_to_bench',inplace=True,ascending=False)
        cluster_ranks.to_csv(output_file,index=False)
        print('--Document clusters similarity to benchmark CSV file created')
        sim_to_bench_sent.append(sents_df.similarity_to_bench.mean())
        print('-----------------------------------------------------------')



    ##################################
    # there is another term to estimate the similarity of each document
    # to the entire benchmarks. columns  are input files and rows are benchmarks
    count_similar_words =np.zeros(len(sim_df.columns))
    for i in range(len(sim_df.columns)) : #input document files
            for j in range(len(sim_df.index)): #bernchmarks
                if(sim_df.iloc[j,i]>doc_sim_threshold ):
                    count_similar_words[i]+=bench_w[j] #1
                
    # normalize the counted words to calculate the similarity of entire documnt to benchmark
    sim_to_bench=count_similar_words/np.sum(bench_w)#len(sim_df.index)

    #####################################



    #put the similarity to becnh for each document     
    proj_df['simimlarity_to_bench']=sim_to_bench
    proj_df['simimlarity_to_bench_sentences']=sim_to_bench_sent

    proj_df[['simimlarity_to_bench_sentences','number_of_cluster']].to_csv(sim_to_bench_file)
    print('Similarity to the benchmark for each input file is written to :',sim_to_bench_file)
    merg_bench_df=(pd.merge(left=bench_df, right=sim_df, left_on='benchmark',right_index=True))
    merg_bench_df.to_csv(sim_to_bench_detail_file,index_label="Seq.")
    print('Similarity to the benchmark for each input file to all distinxt benchmarks :',sim_to_bench_file)
        # creat an numpy array from the weighted centroid vector of all sentences

    bench_class_mediods=np.unique(merg_bench_df['class_mediods_bench'])
    bench_class_tmp=merg_bench_df[merg_bench_df['benchmark'].isin(bench_class_mediods)]
    bench_class_tmp =bench_class_tmp[bench_class_tmp.columns[4:]].copy().reset_index(drop=True)
    bench_class_tmp.loc[len(bench_class_tmp)]=['Total','TOTAL']+list(sim_to_bench)
    bench_class_tmp.to_csv(sim_to_bench_class_file,index=False)
    print('-----------------------------------------------------------')
    print("----End of the process")

    return(True)