import gensim
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import spacy
import pandas as pd
from IPython.core.display import display, HTML
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import sentiwordnet as swn
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import _stop_words
from sklearn.metrics import euclidean_distances
import tensorflow as tf
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.embeddings import WordEmbeddings
import pyLDAvis
import pyLDAvis.gensim_models
from scipy.spatial import distance
import time
from collections import Counter


def entropy(chunk_list):
    """
    Evaluate the entropy of a distribution of chunks.

    :param chunk_list: chunks
    :type chunk_list: list
    :return: entropy of the chunks
    :rtype: float




    """
    freq_list = chunking_count(chunk_list).values()
    return entropy(freq_list, base=2)




def NER(text, df_out=True, viz_out=False):
    """
    Extracts entities. 
    It will return a dataframe in output with all entities inside the document and, if viz_out=True,
    it will visualize the results.
    

    :param text: text to be recognized
    :type text: string
    :param df_out: return dataframe
    :type df_out: bool
    :param viz_out: display visualization in jupyter notebook
    :type viz_out: bool
    :return: entities and types
    :rtype: pandas dataframe

    """

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities = []
    labels = []

    for ent in doc.ents:
        entities.append(ent.text)
        labels.append(ent.label_)

    if viz_out is True:
        display(HTML(spacy.displacy.render(doc, style="ent", page="true")))

    if df_out is True:
        output = pd.DataFrame(columns=["entity", "label"])
        output["entity"] = entities
        output["label"] = labels

    return output
    
def topic_modeling(df, column, num_of_topics=3, algorithm="lda", visualize=False):
    """
    topic_modelling: Topic Modelling is a technique to extract the hidden topics from large volumes of text.
    Latent Dirichlet Allocation(LDA) is a topic modelling technique. In LDA with gensim is employed to create a
    dictionary from the data and then to convert to bag-of-words corpus.
    pLDAvis is designed to help users interpret the topics in a topic model that has been fit to a corpus of text data.
    The package extracts information from a fitted LDA topic model to inform an interactive web-based visualization.


    :param df: DataFrame to be passed
    :param column: Corresponding column of the DatFrame for topic modelling
    :param num_of_topics: Define the number of topics
    :param algorithm: 'lda' - the topic modelling technique
    :param visualize: If 'TRUE' returns the object pyLDAvis.display(lda_display)
    :return: pyLDAvis.display(lda_display) is returned, which is a web-based visualization of topic modelling
    """
    corpus = df[column]
    tokens = [x.split() for x in corpus]
    dictionary = Dictionary(tokens)
    corpus = [dictionary.doc2bow(text) for text in tokens]

    lda = LdaModel(corpus=corpus, num_topics=num_of_topics, id2word=dictionary)
    print(lda.show_topics(num_of_topics))

    if visualize == True:
        lda_display = pyLDAvis.gensim.prepare(
            lda, corpus, dictionary, sort_topics=False
        )
        return (pyLDAvis.display(lda_display)), lda

    return lda


def opinion_polarity(text, lexicon="vadersent"):
    """
    Calculates the polarity of a text based on SentiWordNet or VaderSentiment

    :param text: text to be evaluated
    :type text: string
    :param lexicon: "sentiwordnet", "vadersent"
    :type lexicon: string
    :return: polarity metrics
    :rtype: dict
    
    .. testsetup::

        from Transformations_and_measures import opinion_polarity

    .. doctest::

        >>> textlist = ['I love bananas', 'I hate apples', 'The dog is cute', 'The cat is ugly', 'The car is red', 'The car is not blue']
        >>> for lexicon in ['sentiwordnet', 'vadersent']:
        >>>     for text in textlist:
        >>>         ans = opinion_polarity(text, lexicon)
        >>>         print(lexicon, text, ans)text = 'T1h2e3 q4u5i6c7k} b8r9o0w`n~ f!o@x# $j%u^m&p*e(d o)v-e_r= [t+h,e< l>a/z?y; :d"o{g].'
        sentiwordnet I love bananas {'pos': 0, 'neg': 0, 'polarity': 0}
        sentiwordnet I hate apples {'pos': 0, 'neg': 0, 'polarity': 0}
        sentiwordnet The dog is cute {'pos': 0, 'neg': 0, 'polarity': 0}
        sentiwordnet The cat is ugly {'pos': 0, 'neg': 0, 'polarity': 0}
        sentiwordnet The car is red {'pos': 0, 'neg': 0, 'polarity': 0}
        sentiwordnet The car is not blue {'pos': 0, 'neg': 0, 'polarity': 0}
        vadersent I love bananas {'pos': 0.808, 'neg': 0.0, 'neu': 0.192, 'polarity': 0.6369}
        vadersent I hate apples {'pos': 0.0, 'neg': 0.787, 'neu': 0.213, 'polarity': -0.5719}
        vadersent The dog is cute {'pos': 0.5, 'neg': 0.0, 'neu': 0.5, 'polarity': 0.4588}
        vadersent The cat is ugly {'pos': 0.0, 'neg': 0.524, 'neu': 0.476, 'polarity': -0.5106}
        vadersent The car is red {'pos': 0.0, 'neg': 0.0, 'neu': 1.0, 'polarity': 0.0}
        vadersent The car is not blue {'pos': 0.0, 'neg': 0.0, 'neu': 1.0, 'polarity': 0.0}


    """
    
    if lexicon == "sentiwordnet":
        pos = 0  # sum of positive scores
        neg = 0  # sum of negative scores
        count = 0  # words counter
        tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")  # tokenization of word in the input string
        text = tokenizer.tokenize(text)
        for elem in text:
            try:
                tag = nltk.pos_tag([elem])  # wordnet tag
                convert_tag = {
                    "NN": wn.NOUN,
                    "JJ": wn.ADJ,
                    "VB": wn.VERB,
                    "RB": wn.ADV,
                }  # convert the tag into sentiwordnet tag
                try:
                    tag_swn = convert_tag[tag[0][1][:2]]  # newtag
                except:
                    tag_swn = "a"
                elem_sent = swn.senti_synset(elem + "." + tag_swn + ".01")
                pos = pos + elem_sent.pos_score()
                neg = neg + elem_sent.neg_score()
                count = count + 1
            except Exception as e:
                # print(str(e))
                pass

        if count != 0:
            return {"pos": pos, "neg": neg, "polarity": (pos / count) - (neg / count)}
        else:
            return {"pos": pos, "neg": neg, "polarity": 0}

    # this is executed if type="vadersent"
    # This is another simple but very efficient way to calculate opinion polarity using the library VaderSentiment
    # for additional information on refer to this: https://github.com/nltk/nltk/blob/develop/nltk/sentiment/vader.py
    if lexicon == "vadersent":
        analyzer = SentimentIntensityAnalyzer()  # initializing the sentiment analyzer
        res = analyzer.polarity_scores(text)
        # get the result, having an additional metric for the neutrality as well
        return {
            "pos": res["pos"],
            "neg": res["neg"],
            "neu": res["neu"],
            "polarity": res["compound"],
        }

if __name__ == '__main__':
    textlist = ['I love bananas', 'I hate apples', 'The dog is cute', 'The cat is ugly', 'The car is red', 'The car is not blue']
    for lexicon in ['sentiwordnet', 'vadersent']:
        for text in textlist:
            ans = opinion_polarity(text, lexicon)
            print(lexicon, text, ans)



def pos_tagging(text):
    """
    This function will return a pandas dataframe containing each token of a document with 
    respective part-of-speech and tags.
    
    Parameters:
    text: just pass a text corpus as a string.
    """

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    tokens = []
    pos = []
    tags = []

    for token in doc:
        print(token.text, token.pos_)
        tokens.append(token.text)
        pos.append(token.pos_)
        tags.append(token.tag_)

    df = pd.DataFrame()
    df["tokens"] = tokens
    df["pos"] = pos
    df["tags"] = tags

    return df




def newdoc_similarity(model, word):
    """
    This function will calculate the distance between a new document and each of the docs with which a doc2vec model
    has been trained.
    
    params:
    model: model is a doc2vec Gensim object.
    word: is the word or text string to be evaluated.
    """
    inferred_vector = model.infer_vector([word])

    sims = model.docvecs.most_similar([inferred_vector])

    return sims


class posTagger:

    """
    This class will learn the mapping between the word and the POS they belong to given a corpus, 
    You can use that learning to predict a new word by passing the embeddings;
    
    The class has two methods:
    1. fit: This is where the corpus is parsed, and the learning happens
    2. predict: Once the estimator has learned the patterns you can use this for prediction
    :param training_corpus: Training set from with the estimator will learn, pass a string of your corpus
    :param estimator: Currently ANN is used as the learner, soon a full suite of estimators can be integrated
    :param embedding: Pass the 100 dim embedding in the predict method to get the POS of the word 
    """

    def __init__(self):

        #! Spacy for preprocessing
        self.nlp = spacy.load("en_core_web_sm")
        self.master_dict = {}
        self.df = pd.DataFrame()
        self.df_flair = pd.DataFrame()
        self.epochs = 10

    def fit(self, training_corpus, estimator=None):
        assert type(training_corpus) == str, "Pass a string"

        self.training_corpus = training_corpus
        self.doc = self.nlp(training_corpus)

        #! Get the token and their part of speech
        for token in tqdm((self.doc)):
            if token.pos_ == "PUNCT":
                continue
            elif (
                token.pos_ in ["PRON", "VERB", "NOUN", "ADV", "ADP"]
                and token.text not in self.master_dict.keys()
            ):
                self.master_dict[token.text] = token.pos_

        #! Pass the master_list into the dataframe object
        self.df["Words"] = self.master_dict.keys()
        self.df["POS"] = self.master_dict.values()
        self.df["Words"] = self.df["Words"].apply(lambda x: x.lower())

        #! Get the embeddings for each of the words from the glove emeddings
        self.sentence = Sentence(
            " ".join(list(set(self.training_corpus.lower().split("."))))
        )
        self.glove_embedding = WordEmbeddings("glove")
        self.glove_embedding.embed(self.sentence)

        self.words = [token.text for token in self.sentence]
        self.embedding = [token.embedding for token in self.sentence]

        self.df_flair["Words"] = self.words
        self.df_flair["WordEmbeddings"] = self.embedding
        self.df_flair.drop_duplicates("Words", inplace=True)

        self.master = pd.merge(
            self.df,
            self.df_flair.rename(str, {"embedding": "WordEmbeddings"}),
            how="left",
            on="Words",
        )
        self.master.drop_duplicates(subset="Words", keep="last", inplace=True)
        self.master = pd.concat(
            [
                self.master[["Words", "POS"]],
                self.master["WordEmbeddings"].apply(pd.Series),
            ],
            axis=1,
        )
        self.master = self.master.dropna()

        #! Training the estimator
        X = self.master.iloc[:, 2:]
        y = self.master["POS"]
        self.label_mapper = {"VERB": 0, "PRON": 1, "NOUN": 2, "ADV": 3, "ADP": 4}

        y = y.map(self.label_mapper)

        if estimator == None:

            def artificial_neural_network(X, y, epochs):
                model = tf.keras.Sequential()
                model.add(
                    tf.keras.layers.Dense(
                        units=128, input_shape=([100]), activation=tf.nn.relu
                    )
                )
                model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu))
                model.add(tf.keras.layers.GaussianDropout(0.2))
                model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu))
                model.add(tf.keras.layers.Dropout(0.2))
                model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu))
                model.add(tf.keras.layers.Dropout(0.2))
                model.add(tf.keras.layers.Dense(units=5, activation=tf.nn.sigmoid))
                model.compile(
                    optimizer="adam",
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"],
                )
                model.fit(X, y, batch_size=25, epochs=epochs, validation_split=0.2)
                return model

            self.model = artificial_neural_network(X, y, self.epochs)

    def plot_learning_curve(self):
        fig, ax = plt.subplots(ncols=2, figsize=(18, 7))

        ax[0].plot(self.model.history.history["accuracy"], label="Training Accuracy")
        ax[0].plot(
            self.model.history.history["val_accuracy"], label="Validation Accuracy"
        )
        ax[0].set_title("Learning Curve - Accuracy")
        ax[0].legend()

        ax[1].plot(self.model.history.history["loss"], label="Training Loss")
        ax[1].plot(self.model.history.history["val_loss"], label="Validation Loss")
        ax[1].set_title("Learning Curve - Loss")
        ax[1].legend()

        fig.show()

    def get_embedding(self, word):
        word_embedding = self.master[self.master["Words"] == word].iloc[:, 2:].values
        if len(word_embedding) != 0:
            return word_embedding
        else:
            print("Word not available, Re-train the model with the word!")

    def predict(self, embedding):
        return self.model.predict_classes(embedding)


def neighbor_proba(corpus):
    """
    Note: Corpus should be a list of strings.
    Output: Will return a method, to run it simple use the snippet below,
    
    for i in neighbor_proba(corpus)('machine', count=20):
        print(i)
    'machine': keyword that you want to find the neighbors for.
    'count': return number of rows
    """

    assert isinstance(corpus, list), "Please pass a List of lists"

    l = []
    for i, j in enumerate(corpus):
        try:
            l.append((j, corpus[i + 1]))
        except Exception as e:
            pass

    new_df = pd.DataFrame(l)

    def word_recommendation(word, count=10):
        yield (
            new_df[new_df[0] == word].groupby(1).count().sort_values(0, ascending=False)
            / new_df[new_df[0] == word].groupby(1).count().sum()[0]
        ).reset_index().head(count).rename(str, {1: "Word", 0: "Probability"})

    return word_recommendation


def persistency_index(papers, patents, news, keyword):

    """
    This function will return a persistancy index for a given keyword and for a given list of papers, patents and news.
    
    parameters:
    papers: list of papers (list of strings)
    patents: list of patents (list or strings)
    news: list of news (list of strings)
    keyword: the name of the technology we want to calculate the persistancy index (string)
    
    """

    # checking if input types are correct:

    if type(keyword) != str:
        print("keyword:", "pass a keyword as a string")
    if type(papers) != list:
        print("papers:", "pass a list of strings")
    if type(patents) != list:
        print("patents:", "pass a list of strings")
    if type(news) != list:
        print("news:", "pass a list of strings")
    for i in papers:
        if type(i) != str:
            print("papers:", "pass a list of strings")
    for i in patents:
        if type(i) != str:
            print("patents:", "pass a list of strings")
    for i in news:
        if type(i) != str:
            print("news:", "pass a list of strings")

    # function for counting frequencies of a string in a given text
    def countFreq(pat, txt):
        M = len(pat)
        N = len(txt)
        res = 0

        # A loop to slide pat[] one by one
        for i in range(N - M + 1):

            # For current index i, check
            # for pattern match
            j = 0
            for j in range(M):
                if txt[i + j] != pat[j]:
                    break

            if j == M - 1:
                res += 1
                j = 0
        return res

    # counting the number of times the technology appears into papers
    count_papers = 0
    count_patents = 0
    count_news = 0

    for paper in papers:
        count_papers = count_papers + countFreq(keyword.lower(), paper.lower())

    for patent in patents:
        count_patents = count_patents + countFreq(keyword.lower(), patent.lower())

    for article in news:
        count_news = count_news + countFreq(keyword.lower(), article.lower())

    total = count_papers + count_patents + count_news

    if total != 0:

        persist_papers = count_papers / total
        persist_patents = count_patents / total
        persist_news = count_news / total
    else:
        print("Persistency index: [papers, patents, news]")
        print([0, 0, 0])
        return [0, 0, 0]

    print("Persistency index: [papers, patents, news]")
    print([persist_papers, persist_patents, persist_news])

    return [persist_papers, persist_patents, persist_news]


def word_frequency(df, column, stopwords=None, sort="descending"):
    """
    word_frequency: The function to check for frequency of words. A DataFrame with the corresponding column is passed to
    compute for the word frequencies.

    :param df: DataFrame to be passed
    :param column: Corresponding column of the DataFrame to get word frequencies
    :param stopwords: List of stopwords to be eliminated from the text
    :param sort: To sort the word count in descending order
    :return: DataFrame in the descending order of the word frequencies
    """
    assert isinstance(df, pd.DataFrame), "A DataFrame has to be passed"

    documents = " ".join(list(df[column]))
    documents = [x.strip() for x in documents.split()]

    if stopwords != None:
        words = dict(
            Counter([x.lower() for x in documents if x.lower() not in stopwords])
        )
    else:
        words = dict(Counter([x.lower() for x in documents]))

    word_count_df = pd.DataFrame(
        {"Words": list(words.keys()), "Counts": list(words.values())}
    )

    if sort.lower() == "ascending":
        return word_count_df.sort_values("Counts", ascending=True)
    else:
        return word_count_df.sort_values("Counts", ascending=False)


def word_checker(word, wordlist=None):
    """
    word_checker: The function is invoked when there is a need to check the given word is valid or not.

    :param word: The word to be checked if valid or not
    :param wordlist: If not equal to 'None', check for the validity of the given word
    :return: True if word is in wordlist. False otherwise.
    """

    if wordlist == None:
        fin = open("wordlist.pickle", "rb")
        wordlist = pickle.load(fin)

    return word.lower() in wordlist
