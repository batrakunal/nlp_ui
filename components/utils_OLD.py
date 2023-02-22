from components.wrapperflowW2V import *
import nltk
from components.Commons import *
import requests


def get_embeddings(model, type):
    """
    This function will return a dictionary containing all the embeddings present in the model.
    In particular, keys of the dictionary will be words/strings/chunks/documents, while values will be
    respective vectors.

    Paramters:
    model: pass the model you want to get embeddings from.
    type: supported types "gensim" or "spacy"
    """

    if type == "spacy":

        vocabolario = list(model.vocab.strings)
        output = [model.vocab.get_vector(i) for i in vocabolario]
        embeddings = pd.DataFrame(columns=["vector"], index=vocabolario)
        embeddings["vector"] = output

        if len(embeddings.index[embeddings.index.duplicated()].unique()) is not 0:
            print("Warning: duplicate index.")

        return embeddings["vector"].to_dict()

    if type == "gensim":
        vocab = model.wv.vocab.keys()

        return {i: model.wv[i] for i in vocab}


def words_similarity(word1, word2, embeddings):
    """
    This function will return the similarity between two words.

    Parameters:
    word1: pass word1 as a string
    word2: pass word2 as a string
    embeddings: pass a word embeddings as a dict
    """
    return 1 - distance.cosine(embeddings[word1], embeddings[word2])


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
            print(str(word) + " - Not found")
    # TODO current implementation it is only using average, next step it is to solve this!
    matrix_vecs = np.asanyarray(list_vecs)
    ave_matrix_vecs = matrix_vecs.mean(0)
    ave_embeddings = np.asanyarray(list(embeddings.values())).mean(0)
    return 1 - distance.cosine(ave_matrix_vecs, ave_embeddings)


def word2vec_to_doc2vec(w2v_embedding):
    """

    :param ndarray:
    :return:
    """
    assert type(w2v_embedding) == np.ndarray

    return w2v_embedding.mean(0)


def get_available_models_mongodb(tag):
    client = MongoClient("155.246.39.34", 27017)
    db = client["models"]
    collection = db[tag]
    aslit = [str(id) for id in collection.find().distinct("_id")]
    client.close()
    return aslit


def insert_model_mongodb(tag, model, model_library):
    """

    :param model:
    :return:
    """
    try:
        client = MongoClient("155.246.39.34", 27017)
        db = client["models"]
        fs = gridfs.GridFS(db)
        a = fs.put(model)
        collection = db[tag]
        post = {"_id": tag + "_model", "id": str(a), "model_library": model_library}
        collection.insert_one(post)
        client.close()
        return True
    except:
        return False


def load_embedding_mongodb(tag):
    """

    :param tag:
    :return:
    """
    client = MongoClient("155.246.39.34", 27017)
    tag = tag.replace(" ", "_")
    tag = "ROOM_" + tag
    db = client[tag]
    collection = db["info"]
    fs = gridfs.GridFS(db)
    file = fs.get(ObjectId(collection.find_one({"_id": "embedding"})["obj_id"]))
    binary = file.read()
    client.close()
    adict = read_pickle_obj(binary)
    return adict


def _create_chunks(
    corpus,
    stopwords_after_chunking=True,
    remove_website=True,
    max_ngrams=3,
    min_count=3,
):
    nltk.download("stopwords")
    STOPWORDS = set(nltk.corpus.stopwords.words("english"))
    corpus = re.sub("[^a-zA-Z]+", " ", corpus)
    list_words = corpus.split()
    list_words = [x.lower() for x in list_words]
    list_words = [x for x in list_words if x != ""]
    list_words = [
        x
        for x in list_words
        if x not in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    ]
    list_words = [
        x
        for x in list_words
        if x
        not in [
            ".",
            "!",
            "?",
            "/",
            "\\",
            ",",
            ":",
            ";",
            "[",
            "]",
            "{",
            "}",
            "|",
            "(",
            ")",
        ]
    ]
    if remove_website:
        list_words = [x for x in list_words if x.find("www.") == -1]

    corpus_clean = " ".join(list_words)

    a = phrasing_by_window(corpus_clean, 1, max_ngrams)

    one_gram = a["window_size=1"]
    if stopwords_after_chunking:
        one_gram = [x for x in one_gram if x[0] not in list(STOPWORDS)]
    a["window_size=1"] = one_gram

    b = chunk_freq(a, 1, 3)

    chunks = []
    for key, value in b.items():
        for k, v in value.items():
            if v >= min_count:
                chunks.append(k)

    # transform back to words
    word_chunks = []
    for item in chunks:
        astring = " "
        for i in range(0, len(item), 1):
            astring += item[i] + " "
        word_chunks.append(astring.strip())
    word_chunks = list(set(word_chunks))
    return word_chunks


def _create_embeddings_for_room(
    corpus,
    stopwords_after_chunking=True,
    remove_website=True,
    max_ngrams=3,
    min_count=3,
    embedding_size=100,
    embedding_window=5,
    num_iterations=100000,
):

    word_chunks = _create_chunks(
        corpus, stopwords_after_chunking, remove_website, max_ngrams, min_count
    )

    # create the tensorflow model
    tensor_flow = WrapperFlow(
        word_chunks, embedding_size, embedding_window, num_iterations
    )
    tensor_flow.run()
    embedding = tensor_flow.embeds
    return embedding


# Declare a few helper functions
def read_file_and_save(url, title):
    try:
        r = requests.get(url, stream=True)
        with open(title + ".pdf", "wb") as fd:
            for chunk in r.iter_content(2000):
                fd.write(chunk)
    except Exception as e:
        print(e)


def get_url_download_pdf_read_append(row):
    """

    :param row:
    :return:
    """
    url = row["url"]
    text = None
    title = "todelete"
    if isinstance(url, str):
        if url.find(".pdf") != -1:
            try:
                read_file_and_save(url, title)
                text = extract_text_pdf(title + ".pdf")
                os.remove(title + ".pdf")
                print("DONE")
            except:
                text = "NOT FOUND"
                print("NOT FOUND")
    return text
