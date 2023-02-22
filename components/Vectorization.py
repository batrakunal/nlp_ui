import gensim
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import spacy
import pandas as pd
import numpy as np
import tensorflow as tf
import itertools
from scipy.spatial import distance
import operator
from tqdm import tqdm
import re
import collections
import math
import numpy as np
import random
import tensorflow as tf
import csv
from scipy.spatial import distance
import pickle
import requests
from sklearn.feature_extraction.text import TfidfVectorizer



def tfidf(corpora):
    """
    This function will calculate the tfidf score for words/chunks.
    
    :param corpora: a list of strings where each element is a document(as a string).
    :type corpora: list
    :return: tf-idf for each word
    "rtype: dict
    """
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpora)
    values = X.toarray()[0]
    keys = vectorizer.get_feature_names()
    keys = [i.replace("_"," ") for i in keys]    
    fin_dict = dict(zip(keys, values))
    return fin_dict

class WrapperFlow(object):
    """
    Class responsible of generating embeddings using a Tensorflow wrapper.
    """

    def __init__(self, tokens, embedding_size, window_size, num_iterations):
        """
        Class Constructor
        :param tokens: pass a list of chunks to get the vectors.
        :param embedding_size: number of components of vectors.
        :param window_size: contextual words to a given one to consider during the training.
        :param num_iterations: number of iterations of training (start setting this value low e.g.: 10).
        """
        self.tokens = tokens
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.num_iterations = num_iterations
        self.data, self.count, self.dictionary, self.reverse_dictionary = (
            self.build_dataset()
        )
        self.embeds = None

    def build_dataset(self):

        """
        Builds the following. To understand each of these elements, let us also assume the text "I like to go to school"
        dictionary: maps a string word to an ID (e.g. {I:0, like:1, to:2, go:3, school:4})
        reverse_dictionary: maps an ID to a string word (e.g. {0:I, 1:like, 2:to, 3:go, 4:school}
        count: List of list of (word, frequency) elements (e.g. [(I,1),(like,1),(to,2),(go,1),(school,1)]
        data : Contain the string of text we read, where string words are replaced with word IDs (e.g. [0, 1, 2, 3, 2, 4])
        It also introduces an additional special token UNK to denote rare words to are too rare to make use of.

        Parameters:
        words: pass a list of words/tokens/chunks with which you want to train a w2v model.
        """

        vocabulary_size = len(set(self.tokens))

        count = [["UNK", -1]]
        # Gets only the vocabulary_size most common words as the vocabulary
        # All the other words will be replaced with UNK token
        count.extend(collections.Counter(self.tokens).most_common(vocabulary_size - 1))
        dictionary = dict()

        # Create an ID for each word by giving the current length of the dictionary
        # And adding that item to the dictionary
        for word, _ in count:
            dictionary[word] = len(dictionary)

        data = list()
        unk_count = 0
        # Traverse through all the text we have and produce a list
        # where each element corresponds to the ID of the word found at that index
        for word in self.tokens:
            # If word is in the dictionary use the word ID,
            # else use the ID of the special token "UNK"
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count = unk_count + 1
            data.append(index)

        # update the count variable with the number of UNK occurences
        count[0][1] = unk_count

        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        # Make sure the dictionary is of size of the vocabulary
        assert len(dictionary) == vocabulary_size

        return data, count, dictionary, reverse_dictionary

    def generate_batch_skip_gram(self, batch_size, window_size):
        """
        this function will be used by the train_w2v function to split data in batches for the iterations.
        It generates a batch or target words (batch) and a batch of corresponding context words (labels).
        It reads 2*window_size+1 words at a time (called a span) and create 2*window_size datapoints in a single span.
        The function continue in this manner until batch_size datapoints are created.
        """

        # data_index is updated by 1 everytime we read a data point
        data_index = 0

        # two numpy arras to hold target words (batch)
        # and context words (labels)
        batch = np.ndarray(shape=batch_size, dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

        # span defines the total window size, where
        # data we consider at an instance looks as follows.
        # [ skip_window target skip_window ]
        span = 2 * window_size + 1

        # The buffer holds the data contained within the span
        buffer = collections.deque(maxlen=span)

        # Fill the buffer and update the data_index
        for _ in range(span):
            buffer.append(self.data[data_index])
            data_index = (data_index + 1) % len(self.data)

        # This is the number of context words we sample for a single target word
        num_samples = 2 * window_size

        # We break the batch reading into two for loops
        # The inner for loop fills in the batch and labels with
        # num_samples data points using data contained withing the span
        # The outper for loop repeat this for batch_size//num_samples times
        # to produce a full batch
        for i in range(batch_size // num_samples):
            k = 0
            # avoid the target word itself as a prediction
            # fill in batch and label numpy arrays
            for j in list(range(window_size)) + list(
                range(window_size + 1, 2 * window_size + 1)
            ):
                batch[i * num_samples + k] = buffer[window_size]
                labels[i * num_samples + k, 0] = buffer[j]
                k += 1

            # Everytime we read num_samples data points,
            # we have created the maximum number of datapoints possible
            # withing a single span, so we need to move the span by 1
            # to create a fresh new span
            buffer.append(self.data[data_index])
            data_index = (data_index + 1) % len(self.data)

        return batch, labels

    def train_w2v(
        self,
        embedding_size,
        window_size,
        num_iterations,
        num_sampled=32,
        save_embeddings_pickle=False,
        save_skip_losses=False,
    ):
        """
        This function will train a word2vec model using the skipgram algorithm implementation in Tensorflow (1.4 verified).
        It will return a dict with embeddings as output.

        Parameters:
        embedding_size: Dimension of the embedding vector.
        window_size: How many words to consider left and right.
        num_iteratons: number of epochs for the training (keep it low for short text corpora, otherwise you will receive an error)
        """
        num_steps = num_iterations
        batch_size = embedding_size
        valid_size = 16  # Random set of words to evaluate similarity on.
        # We sample valid datapoints randomly from a large window without always being deterministic
        valid_window = 50
        vocabulary_size = len(set(self.tokens))
        # When selecting valid examples, we select some of the most frequent words as well as
        # some moderately rare words as well
        valid_examples = np.array(random.sample(range(valid_window), valid_size))
        valid_examples = np.append(
            valid_examples,
            random.sample(range(1000, 1000 + valid_window), valid_size),
            axis=0,
        )
        tf.reset_default_graph()

        # Training input data (target word IDs).
        train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
        # Training input label data (context word IDs)
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        # Validation input data, we don't need a placeholder
        # as we have already defined the IDs of the words selected
        # as validation data
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # Embedding layer, contains the word embeddings
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
        )

        # Softmax Weights and Biases
        softmax_weights = tf.Variable(
            tf.truncated_normal(
                [vocabulary_size, embedding_size],
                stddev=0.5 / math.sqrt(embedding_size),
            )
        )
        softmax_biases = tf.Variable(tf.random_uniform([vocabulary_size], 0.0, 0.01))

        # Model.
        # Look up embeddings for a batch of inputs.
        embed = tf.nn.embedding_lookup(embeddings, train_dataset)

        # Compute the softmax loss, using a sample of the negative labels each time.
        loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(
                weights=softmax_weights,
                biases=softmax_biases,
                inputs=embed,
                labels=train_labels,
                num_sampled=num_sampled,
                num_classes=vocabulary_size,
            )
        )

        # Compute the similarity between minibatch examples and all embeddings.
        # We use the cosine distance:
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

        # Optimizer.
        optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

        skip_losses = []
        # ConfigProto is a way of providing various configuration settings
        # required to execute the graph
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
            # Initialize the variables in the graph
            tf.global_variables_initializer().run()
            print("Initialized")
            average_loss = 0

            # Train the Word2vec model for num_step iterations
            for step in range(num_steps):

                # Generate a single batch of data
                batch_data, batch_labels = self.generate_batch_skip_gram(
                    batch_size, window_size
                )

                # Populate the feed_dict and run the optimizer (minimize loss)
                # and compute the loss
                feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
                _, l = session.run([optimizer, loss], feed_dict=feed_dict)

                # Update the average loss variable
                average_loss += l

                if (step + 1) % 2000 == 0:
                    if step > 0:
                        average_loss = average_loss / 2000

                    skip_losses.append(average_loss)
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print("Average loss at step %d: %f" % (step + 1, average_loss))
                    average_loss = 0

                # Evaluating validation set word similarities
                if (step + 1) % 10000 == 0:
                    sim = similarity.eval()
                    # Here we compute the top_k closest words for a given validation word
                    # in terms of the cosine distance
                    # We do this for all the words in the validation set
                    # Note: This is an expensive step
                    for i in range(valid_size):
                        valid_word = self.reverse_dictionary[valid_examples[i]]
                        top_k = 8  # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1 : top_k + 1]
                        log = "Nearest to %s:" % valid_word
                        for k in range(top_k):
                            close_word = self.reverse_dictionary[nearest[k]]
                            log = "%s %s," % (log, close_word)
                        print(log)
            skip_gram_final_embeddings = normalized_embeddings.eval()

        # We will save the word vectors learned and the loss over time
        # as this information is required later for comparisons
        if save_skip_losses:
            with open("skip_losses.csv", "wt") as f:
                writer = csv.writer(f, delimiter=",")
                writer.writerow(skip_losses)

        my_embeddings = {}

        for i in range(0, len(skip_gram_final_embeddings)):
            my_embeddings[self.reverse_dictionary[i]] = skip_gram_final_embeddings[i]
        len(my_embeddings)
        if save_embeddings_pickle:
            with open("embeddings.pickle", "wb") as file:
                pickle.dump(my_embeddings, file, protocol=pickle.HIGHEST_PROTOCOL)

        return my_embeddings

    @staticmethod
    def words_similarity(word1, word2, embeddings):
        """
        This function will return the similarity between two words.

        Parameters:
        word1: pass word1 as a string
        word2: pass word2 as a string
        embeddings: pass a word embeddings as a dict
        """
        return 1 - distance.cosine(embeddings[word1], embeddings[word2])

    def run(self):
        self.embeds = self.train_w2v(
            embedding_size=self.embedding_size,
            window_size=self.window_size,
            num_iterations=self.num_iterations,
        )


class word2vec:
    """
    This class will provide a way for training a word2vec model using TensorFlow.
    
    Parameters:
    
    :param corpus: list of lists of tokens to be passed 
    :param window: contextual words to a given one to be considered during the training process.
    :param embedding_dim: number of components each vector has.
    :param n_iter: Number of iterations of the training.
    :param verbose: If > 0, prints the loss
    
    :return: Trained w2v model, use query() to pull similar words
    """

    def __init__(self, corpus, window, embedding_dim, n_iter, verbose):
            self.corpus = corpus
            self.word_list = list(itertools.chain(corpus))
            #self.word_list = list(set(corpus))
            self.window = window
            self.embedding_dim = embedding_dim
            self.n_iter = n_iter
            self.verbose = verbose

    def preprocess(self):
            self.words = set(
                    [word for word in self.word_list]
            )  ## Insert a if statement to check the word before appending
            self.int2word, self.word2int = {}, {}

            #! Create a look-up dictionary
            for idx, word in enumerate(self.words):
                    self.word2int[word] = idx
                    self.int2word[idx] = word

            self.data = []
            for sentence in self.corpus:
                    for word_index, word in enumerate(sentence):
                            for nb_word in sentence[
                                    max(word_index - self.window, 0) : min(
                                            word_index + self.window, len(sentence)
                                    )
                                    + 1
                            ]:
                                    if nb_word != word:
                                            self.data.append([word, nb_word])

            def one_hot_encode(data_idx, vocab_size):
                    ohe = np.zeros(vocab_size)
                    ohe[data_idx] = 1
                    return ohe

            self.X_train, self.y_train = [], []

            for token in self.data:
                    self.X_train.append(
                            one_hot_encode(self.word2int[token[0]], len(self.words))
                    )
                    self.y_train.append(
                            one_hot_encode(self.word2int[token[1]], len(self.words))
                    )

            self.X_train = np.asarray(self.X_train)
            self.y_train = np.asarray(self.y_train)

    def build(self):
            self.x = tf.placeholder(tf.float32, shape=(None, len(self.words)))
            self.y_label = tf.placeholder(tf.float32, shape=(None, len(self.words)))

            self.W1 = tf.Variable(tf.random_normal([len(self.words), self.embedding_dim]))
            self.b1 = tf.Variable(tf.random_normal([self.embedding_dim]))  # bias
            self.hidden_representation = tf.add(tf.matmul(self.x, self.W1), self.b1)

            self.W2 = tf.Variable(tf.random_normal([self.embedding_dim, len(self.words)]))
            self.b2 = tf.Variable(tf.random_normal([len(self.words)]))
            self.prediction = tf.nn.softmax(
                    tf.add(tf.matmul(self.hidden_representation, self.W2), self.b2)
            )

    def run(self):
            self.sess = tf.Session()
            self.init = tf.global_variables_initializer()
            self.sess.run(self.init)

            self.cross_entropy_loss = tf.reduce_mean(
                    -tf.reduce_sum(
                            self.y_label * tf.log(self.prediction), reduction_indices=[1]
                    )
            )
            self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(
                    self.cross_entropy_loss
            )
            for _ in tqdm(range(self.n_iter)):
                    self.sess.run(
                            self.train_step,
                            feed_dict={self.x: self.X_train, self.y_label: self.y_train},
                    )
                    if self.verbose > 0:
                            print(
                                    "loss is : ",
                                    self.sess.run(
                                            self.cross_entropy_loss,
                                            feed_dict={self.x: self.X_train, self.y_label: self.y_train},
                                    ),
                            )

            self.vectors = self.sess.run(self.W1 + self.b1)
            self.word_embeddings = {}
            for words in self.word2int.keys():
                    self.word_embeddings[words] = self.vectors[self.word2int[words]].tolist()

    def query(self, word):
            self.most_similar = {}
            for k, v in self.word_embeddings.items():
                    dst = distance.euclidean(self.word_embeddings[word], v)
                    self.most_similar[k] = dst
            self.most_similar = sorted(
                    self.most_similar.items(), key=operator.itemgetter(1)
            )
            return self.most_similar

def w2v(df, column, size=300, window=5, min_count=3, phrases=False):
    """
    w2v: Word2vec in gensim library is used to produce word vectors with deep learning which includes "skip-gram and CBOW
    models", using either hierarchical softmax or negative sampling. A DataFrame with a corresponding column is passed to
    create word vectors.
    
    Parameters:
    
    df: The DataFrame to be passed containing a column with each row as a string (text).
    column: The column containing the text to generate corresponding word vectors
    size: size of the embeddings (number of components of a vector)
    window: number of words to consider contextual to a given one during the training.
    min_count: minimum count of a word to be considered in the training process.
    save: set True if want to save the model in the working directory.
    model_name: pass a string with the file name of the model (default "model.bin").
    phrases: If 'TRUE', returns the model
    """
    assert isinstance(df, pd.DataFrame), "A DataFrame has to be passed"

    corpus = list(df[column])
    tokens = [x.split() for x in corpus]

    if phrases:
        bigrams = gensim.models.Phrases(tokens)
        texts = [bigrams[line] for line in tokens]
        model = gensim.models.Word2Vec(texts, size=size, window=window, min_count=min_count)
    else:
        model = gensim.models.Word2Vec(tokens, size=size, window=window, min_count=min_count)
    return model


def base_w2v(list_of_chunks, size, window, min_count, iterations, selected_seed, save=False, model_name="model"):
    """
    This function will return a word2vec model from a reference corpus using the Gensim way.
    
    Parameters:
    reference_corpus: pass a a list of chunks.
    size: size of the embeddings (number of components of a vector)
    window: number of words to consider contextual to a given one during the training.
    min_count: minimum count of a word to be considered in the training process.
    interations: number of iterations for the training.
    selected_seed: seed number for the training.
    save: set True if want to save the model in the working directory.
    model_name: pass a string with the file name of the model (default "model.bin").
    """
    
    list_of_chunks = [list_of_chunks]

    model = gensim.models.Word2Vec(list_of_chunks, size=size, window=window, 
                                   min_count=min_count, workers=-1, iter=iterations, seed=selected_seed)

    if save == True:
        model.wv.save_word2vec_format(model_name + ".bin", binary=True)

    return model


def plain_w2v_embeddings(model, type="g"):
    """
    Create a dictionary containing all the embeddings present in the model.
    In particular, keys of the dictionary will be words/strings/chunks/documents, while values will be
    respective vectors.

    Paramters:
    :param model: model
    :type model: Word2Vec model
    :param type: supported types "g" stands for "gensim" or "s" that stands for "spacy"
    :type type: string
    :return: embeddings
    :rtype: dict
    """

    if type == "s":

        voc = list(model.vocab.strings)
        output = [model.vocab.get_vector(i) for i in voc]
        embeddings = pd.DataFrame(columns=["vector"], index=voc)
        embeddings["vector"] = output

        if len(embeddings.index[embeddings.index.duplicated()].unique()) is not 0:
            print("Warning: duplicate index.")

        return embeddings["vector"].to_dict()

    if type == "g":

        vocab = model.wv.vocab.keys()

        return {i: model.wv[i] for i in vocab}


def pretrained_word2vec_model(model="google", limit=1000000):
    """
    This function loads a pre-trained word2vec model. The option 'google' loads the pre trained 3 millions words
    corpus trained by google.
    :param model: basestring Defines the model to be loaded (at the moment only google-news model is available)
    :return: gensim.model
    """
    assert isinstance(model, str), "Please pass a string"

    if model == "google":
        url = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
        print("Hang in there, it will take a while. After all, it is a 3 million word model!")
        return gensim.models.KeyedVectors.load_word2vec_format(url, binary=True, limit=limit)
    else:
        print("Specify a valid model name")


def atomic_doc2vec(rooms_text, vector_size, window, min_count):
    """
    This function will train a Doc2Vec model for each of the docs we want to pass.
    
    Parameters:
    rooms_text: just pass a list where each element is the corpus of a specific room.
    vector_size: size of the embeddings (number of components of a vector)
    window: number of words to consider contextual to a given one during the training.
    min_count: minimum count of a word to be considered in the training process.
    """

    print("you have", len(rooms_text), "docs available!")

    rooms_tokens = [
        i.split(" ") for i in rooms_text
    ]  # tokenize (of course clean using text cleaner component)

    documents = [
        TaggedDocument(doc, ["doc_" + str(i)]) for i, doc in enumerate(rooms_tokens)
    ]

    model = Doc2Vec(
        documents,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
    )

    return model


def plain_doc_embeddings(model):
    """
    This function will extract plain embeddings of documents from a doc2vec model.
    
    parameters:
    model: just pass the doc2vec model.
    """

    doc_emb = {}

    for i in range(0, len(model.docvecs)):
        doc_emb["doc_" + str(i)] = model.docvecs[i]

    return doc_emb


class doc2vec:
    def __init__(self, docs):
        assert isinstance(docs, list), "Please pass a list of strings"
        self.docs = docs

    def preprocess(self):
        pass

    def build(self, min_count, window=5, vector_size=300, workers=4):

        self.tokens = [
            i.split(" ") for i in self.docs
        ]  # tokenize (of course clean using text cleaner component)
        documents = [
            TaggedDocument(doc, ["doc_" + str(i)]) for i, doc in enumerate(self.tokens)
        ]
        self.model = Doc2Vec(
            documents,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
        )
        return self.model

    def infer_embeddings(self, doc):
        assert isinstance(doc, str), "Please pass a string"

        infered_vector = self.model.infer_vector([doc])
        return infered_vector

    def extract_embeddings(self):

        doc_emb = {}
        for i in range(0, len(self.model.docvecs)):
            doc_emb["doc_" + str(i)] = self.model.docvecs[i]
        return doc_emb

    def newdoc_similarity(self, word):
        inferred_vector = self.model.infer_vector([word])
        sims = self.model.docvecs.most_similar([inferred_vector])
        return sims

    @staticmethod
    def save_model(model, model_name):
        model.save(model_name + ".model")
        print("Model Saved")

    @staticmethod
    def load_model(model):
        model = Doc2vec.load(model)
        return model


def googlenews_model():
    """
    This function will point at the url to download the 'GoogleNews-vectors-negative300.bin'.
    This is a pre-trained word2vec model with millions of News.
    """

    url = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
    file_name = "GoogleNews-vectors-negative300.bin.gz"

    r = requests.get(url, stream=True)
    fileSize = int(r.headers["Content-Length"])
    downloaded = 0
    chunkSize = 1024
    bars = int(fileSize / chunkSize)
    with open(file_name, "wb") as f:
        for chunk in tqdm(
            r.iter_content(chunk_size=chunkSize),
            total=bars,
            unit="KB",
            desc=file_name,
            leave=True,
        ):
            f.write(chunk)
            downloaded += chunkSize
            prog = downloaded * 100 / fileSize
            
           
def smart_procrustes_align_gensim(base_embed, other_embed, words=None):
    """Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
    Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.
        (With help from William. Thank you!)
    First, intersect the vocabularies (see `intersection_align_gensim` documentation).
    Then do the alignment on the other_embed model.
    Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
    Return other_embed.
    If `words` is set, intersect the two models' vocabulary with the vocabulary in words (see `intersection_align_gensim` documentation).
    """
    
    # patch by Richard So [https://twitter.com/richardjeanso) (thanks!) to update this code for new version of gensim
    base_embed.init_sims()
    other_embed.init_sims()

    # make sure vocabulary and indices are aligned
    in_base_embed, in_other_embed = intersection_align_gensim(base_embed, other_embed, words=words)

    # get the embedding matrices
    base_vecs = in_base_embed.syn0norm
    other_vecs = in_other_embed.syn0norm

    # just a matrix dot product with numpy
    m = other_vecs.T.dot(base_vecs) 
    # SVD method from numpy
    u, _, v = np.linalg.svd(m)
    # another matrix operation
    ortho = u.dot(v) 
    # Replace original array with modified one
    # i.e. multiplying the embedding matrix (syn0norm)by "ortho"
    other_embed.syn0norm = other_embed.syn0 = (other_embed.syn0norm).dot(ortho)
    return other_embed
    
def intersection_align_gensim(m1,m2, words=None):
    """
    Intersect two gensim word2vec models, m1 and m2.
    Only the shared vocabulary between them is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
    These indices correspond to the new syn0 and syn0norm objects in both gensim models:
        -- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
        -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
    The .vocab dictionary is also updated for each model, preserving the count but updating the index.
    """

    # Get the vocab for each model
    vocab_m1 = set(m1.wv.vocab.keys())
    vocab_m2 = set(m2.wv.vocab.keys())

    # Find the common vocabulary
    common_vocab = vocab_m1&vocab_m2
    if words: common_vocab&=set(words)

    # If no alignment necessary because vocab is identical...
    if not vocab_m1-common_vocab and not vocab_m2-common_vocab:
        return (m1,m2)

    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(key=lambda w: m1.wv.vocab[w].count + m2.wv.vocab[w].count,reverse=True)

    # Then for each model...
    for m in [m1,m2]:
        # Replace old syn0norm array with new one (with common vocab)
        indices = [m.wv.vocab[w].index for w in common_vocab]
        old_arr = m.wv.syn0norm
        new_arr = np.array([old_arr[index] for index in indices])
        m.syn0norm = m.syn0 = new_arr

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
        m.index2word = common_vocab
        old_vocab = m.wv.vocab
        new_vocab = {}
        for new_index,word in enumerate(common_vocab):
            old_vocab_obj=old_vocab[word]
            new_vocab[word] = gensim.models.word2vec.Vocab(index=new_index, count=old_vocab_obj.count)
        m.vocab = new_vocab

    return (m1,m2)
