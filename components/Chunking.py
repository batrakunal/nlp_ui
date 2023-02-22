import spacy
from spacy.lang.en import English
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.embeddings import WordEmbeddings
from tqdm import tqdm
import pandas as pd
import gensim
from collections import Counter


def chunking(text):
    """
    Splits a test into meaningful chunks.

    :param text: text
    :type text: string
    :return: chunks
    :rtype: list

    """
    l1 = np_chunking(text)  
    # l2 = chunking_NER(text)
    # try:
    #     l3 = chunking_fair(text, tags=False)
    #     l4 = chunking_soa(text, tags=False)
    # except:
    #     pass
    return l1


def chunking_count(chunk_list):
    """
    Counts chunk frequency in list.

    :param chunk_list: chunks
    :type chunk_list: list
    :return: chunks with frequency
    :rtype: Counter dict
    """
    return Counter(chunk_list)


def chunking_count_most_common(chunk_list, n=10):
    """
    Most common chunks in list.

    :param chunk_list: chunks
    :type chunk_list: list
    :param n: number of most common chunks
    :type n: int
    :return: n chunks with largest frequency
    :rtype: list
    """
    return Counter(chunk_list).most_common(n)


def phrasing_by_window(corpus, chunk_length, up_to=None):
    """
    Creates word chunks of predetermined size ranging from "chunk_length" to "up_to".
    
    :param corpus: text
    :type corpus: string
    :param chunk_length: length of chunks
    :type chunk_length: int
    :param up_to: maximum length of chunks
    :type up_to: int
    :return: chunks in the form {'chunk_length=n':[tuple1, tuple2, ...]}
    :rtype: dict

    .. testsetup::

        from Chunking import phrasing_by_window

    .. doctest::

        >>> corpus = 'The quick brown fox jumped over the lazy dog.'
        >>> phrasing_by_window(corpus, 2)
        {'chunk_length=2': [('The', 'quick'), ('quick', 'brown'), ('brown', 'fox'), ('fox', 'jumped'), ('jumped', 'over'), ('over', 'the'), ('the', 'lazy'), ('lazy', 'dog.')]}

        >>> corpus = 'The quick brown fox jumped over the lazy dog.'
        >>> phrasing_by_window(corpus, 3, 5)
        {'chunk_length=3': [('The', 'quick', 'brown'), ('quick', 'brown', 'fox'), ('brown', 'fox', 'jumped'), ('fox', 'jumped', 'over'), ('jumped', 'over', 'the'), ('over', 'the', 'lazy'), ('the', 'lazy', 'dog.')], 'chunk_length=4': [('The', 'quick', 'brown', 'fox'), ('quick', 'brown', 'fox', 'jumped'), ('brown', 'fox', 'jumped', 'over'), ('fox', 'jumped', 'over', 'the'), ('jumped', 'over', 'the', 'lazy'), ('over', 'the', 'lazy', 'dog.')], 'chunk_length=5': [('The', 'quick', 'brown', 'fox', 'jumped'), ('quick', 'brown', 'fox', 'jumped', 'over'), ('brown', 'fox', 'jumped', 'over', 'the'), ('fox', 'jumped', 'over', 'the', 'lazy'), ('jumped', 'over', 'the', 'lazy', 'dog.')]}

    """
    if up_to == None:
        up_to = chunk_length
    chunk_dict = {}
    for i in range(chunk_length, up_to + 1):
        in_list = corpus.split(" ")
        chunks = list(zip(*[in_list[j:] for j in range(i)]))
        chunk_dict["chunk_length=" + str(i)] = chunks
    return chunk_dict


def chunk_frequency(chunks_dict, chunk_length, up_to=None):
    """
    Calculates the frequency with which each chunk has occurred within the corpus.

    :param chunks_dict: 
    :type chunks_dict: dict
    :param chunk_length: length of chunks
    :type chunk_length: int
    :param up_to: maximum length of chunks
    :type up_to: int
    :return: frequency of each chunk
    :rtype: dict

    .. testsetup::

        from Chunking import phrasing_by_window
        corpus = 'The quick brown fox jumped over the lazy dog.'
        chunks_dict = phrasing_by_window(corpus, 2)
        from Chunking import chunk_frequency

    .. doctest::

        >>> chunk_frequency(chunks_dict, 2)
        {'chunk_length=2': {('The', 'quick'): 1, ('quick', 'brown'): 1, ('brown', 'fox'): 1, ('fox', 'jumped'): 1, ('jumped', 'over'): 1, ('over', 'the'): 1, ('the', 'lazy'): 1, ('lazy', 'dog.'): 1}}
    """
    if up_to == None:
        up_to = chunk_length
    total_chunks_count = {}
    for i in range(chunk_length, up_to + 1):
        chunks_count = {}
        for chunk in chunks_dict["chunk_length=" + str(i)]:
            if chunk not in chunks_count:
                chunks_count[chunk] = 1
            else:
                chunks_count[chunk] = chunks_count[chunk] + 1
        total_chunks_count["chunk_length=" + str(i)] = chunks_count
    return total_chunks_count


def chunks_toplist(chunk_frequency, chunk_length, up_to=None):
    """
    Evaluates the most frequent chunks for each chunk length.

    :param chunk_frequency: frequency of each chunk
    :type chunk_frequency: dict
    :param chunk_length: length of chunks
    :type chunk_length: int
    :param up_to: maximum length of chunks
    :type up_to: int
    :return: most frequent chunk for each length
    :rtype: list

    """
    if up_to == None:
        up_to = chunk_length
    total_top_chunks = {}
    for i in range(chunk_length, up_to + 1):
        max_value = max(chunk_frequency["chunk_length=" + str(i)].values())
        if max_value == 1:
            total_top_chunks["chunk_length=" + str(i)] = []
        else:
            top_chunks = []
            for chunk in chunk_frequency["chunk_length=" + str(i)]:
                if max_value == chunk_frequency["chunk_length=" + str(i)][chunk]:
                    top_chunks.append(chunk)
                    total_top_chunks["chunk_length=" + str(i)] = top_chunks
    return total_top_chunks


# def new_corpora_dict(chunks_top, corpus_raw, chunk_length, up_to):
#     """
#     Finally, the following function "new_corpora_dict(chunks_top,corpus_raw,window_size,up_top)"
#     is going to 
#     Substitute the top chunks in the original corpus (named as "corpus_raw" in the function's inputs)
#     with a unique string with "_" instead of " " between words of the chunk.
    
#     params:
    
#     chunks_top: pass the output of chunks_toplist function (dictionary).
#     corpus_raw: pass the raw text as a string.
#     window_size: set the minimum number of tokens for creating a chunk.
#     up_to: maximum number of tokens for creating a chunk.    


#     Substitutes the top chunks in the original corpus
#     (named as "corpus_raw" in the function's inputs)
#     with a unique string with "_" instead of " " between words of the chunk.
#     """
#     new_corpora = {}
#     for i in range(chunk_length, up_to + 1):
#         corpus = ""
#         if len(chunks_top["chunk_length=" + str(i)]) == 0:
#             corpus = corpus_raw
#             new_corpora["chunk_length=" + str(i)] = corpus
#         else:
#             for chunk in chunks_top["chunk_length=" + str(i)]:
#                 if corpus == "":
#                     corpus = corpus_raw.replace(" ".join(chunk), "_".join(chunk))
#                 else:
#                     corpus = corpus.replace(" ".join(chunk), "_".join(chunk))
#             new_corpora["chunk_length=" + str(i)] = corpus
#     return new_corpora


def chunking_fair(text, tags=False):
    """
    Chunks a text by using a combination of named entity recognition and chunking_soa.
    The output of the function is a list of chunks.
    The chunking method does not remove stopwords before the processing. You will need to clean stopwords after 
    by using a text cleaner. This method has been designed for structured text liks book/news/papers/patents corpora.
    
    Parameters:
    text: just pass a string of the text (cleaned of punctuantion with the "textcleaner component", 
          but without removing stopwords please!)
    """

    nlp = English()
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)
    doc = nlp(text)
    doc = [x.text for x in doc.sents]

    final_chunks = []

    # load the chunk tagger
    tagger1 = SequenceTagger.load("chunk-fast")

    # load the NER tagger
    tagger2 = SequenceTagger.load("ner")

    for text in tqdm(doc):

        assert type(text) == str, print("Wrong input!")
        text = text.replace("<", "")
        text = text.replace(">", "")

        # print(" chunking_soa + NER.....")

        sentence = Sentence(text + " .")

        # run chunk over sentence
        tagger1.predict(sentence)

        # load the NER tagger
        tagger2.predict(sentence)

        tagged = sentence.to_tagged_string()

        first_split = tagged.split(">")

        chunked = []

        for i in first_split:
            tok = i.strip().split(" <")
            chunked.append(tok)

        df = pd.DataFrame(chunked, columns=["tokens", "tags"])

        chunks = []

        stringa = ""

        for i in range(0, len(df) - 1):

            if "/" not in df["tags"][i]:

                if "B" in df["tags"][i]:
                    stringa = stringa + "|||" + df["tokens"][i]
                if "S" in df["tags"][i]:
                    stringa = stringa + "|||" + df["tokens"][i]
                if "I" in df["tags"][i]:
                    stringa = stringa + " " + df["tokens"][i]
                if "E" in df["tags"][i]:
                    stringa = stringa + " " + df["tokens"][i]

            if "/" in df["tags"][i]:
                if "/B" in df["tags"][i]:
                    stringa = stringa + "|||" + df["tokens"][i]
                if "/S" in df["tags"][i]:
                    stringa = stringa + "|||" + df["tokens"][i]
                if "/I" in df["tags"][i]:
                    stringa = stringa + " " + df["tokens"][i]
                if "/E" in df["tags"][i]:
                    stringa = stringa + " " + df["tokens"][i]

        chunked_text = stringa.split("|||")

        chunklist = [item.replace(" ", " ") for item in chunked_text][1:]
        for i in chunklist:
            final_chunks.append(i)
        # print(chunklist)

    if tags is True:
        return df
    else:
        return final_chunks


def chunking_soa(text, tags=False):
    """
    Chunks a text using 

    The output of this function is a list of chunks.
    The chunking method does not remove stopwords before the processing. You will need to clean stopwords after
    by using a text cleaner. This method has been designed for structured text liks book/news/papers/patents corpora.


    :param text: text
    :type text: string
    :param tags: tags
    :rtype tags: bool
    :return: chunks
    :rtype: list

    .. testsetup::

        from Chunking import chunking_soa

    .. doctest::
    
        >>> text = 'It was 1988 and the White House was in Washington D.C. as it had always been.'
        >>> chunking_soa(text)
        None




    """
    nlp = English()
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)
    doc = nlp(text)
    doc = [x.text for x in doc.sents]

    final_chunks = []

    # load the chunk tagger
    tagger = SequenceTagger.load("chunk-fast")

    for text in tqdm(doc):
        assert type(text) == str, print("Wrong input!")
        text = text.replace("<", "")
        text = text.replace(">", "")

        sentence = Sentence(text)

        # run chunk over sentence
        tagger.predict(sentence)
        # tagged sentence
        tagged = sentence.to_tagged_string()
        # splitting chunks and tags
        first_split = tagged.split(">")
        # creating df
        chunked = []
        for i in first_split:
            tok = i.strip().split(" <")
            chunked.append(tok)

        df = pd.DataFrame(chunked, columns=["tokens", "tags"])

        chunks = []
        temp_chunks = []

        for row in range(0, len(df) - 1):
            if df["tags"][row] is not None:
                if df["tags"][row].startswith("B"):
                    temp_chunks.append(df["tokens"][row])

                if df["tags"][row].startswith("I"):
                    temp_chunks.append(df["tokens"][row])

                if df["tags"][row].startswith("E"):
                    chunks.append(temp_chunks + [df["tokens"][row]])
                    temp_chunks = []

                if df["tags"][row].startswith("S"):
                    chunks.append([df["tokens"][row]])

                else:
                    pass

        chunklist = [" ".join(i) for i in chunks]
        for i in chunklist:
            final_chunks.append(i)
        # print(chunklist)

    if tags is True:
        return df
    else:
        return final_chunks


def np_chunking(text):
    """
    Chunks a text using noun phrase chunking.

    :param text: text
    :type text: string
    :return: chunks
    :rtype: list

    .. testsetup::

        from Chunking import np_chunking

    .. doctest::
    
        >>> text = 'It was 1988 and the White House was in Washington D.C. as it had always been.'
        >>> np_chunking(text)
        ['It', 'the White House', 'Washington D.C.', 'it']

    """
    max_length = 999999
    nlp = spacy.load("en_core_web_sm")

    def subdivision(x, l):
        r = x / l
        a = int(r)
        final = [l] * a
        final.append(x - sum(final))
        return final

    if len(text) < max_length:
        doc = nlp(text)
        return [chunk.text for chunk in doc.noun_chunks]
    else:
        chunks = []
        sliced = subdivision(len(text), max_length)
        for s in sliced:
            doc = nlp(text[:s])
            for chunk in doc.noun_chunks:
                chunks.append(chunk.text)
        return chunks


def chunking_NER(text):
    """
    Chunks a text using named entity recognition.

    :param text: text
    :type text: string
    :return: chunks
    :rtype: list

    .. testsetup::

        from Chunking import chunking_NER

    .. doctest::
    
        >>> text = 'It was 1988 and the White House was in Washington D.C. as it had always been.'
        >>> chunking_NER(text)
        ['1988', 'the White House', 'Washington D.C.']

    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    chunks = [ent.text for ent in doc.ents]
    return chunks


def dependency_parser(corpus):
    """
    This function will parse a text using spacy implementation.
    params:
    corpus: pass a corpus as a string.
    """
    assert type(corpus) == str
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(corpus)
    ROOT = str(corpus).split()[[x.dep_ for x in doc].index("ROOT")]
    text = corpus
    return [(x.text, x.dep_) for x in doc]


def phrases(df, column):
    """
    Detects common phrase, multi-word expressions or word n-grams, from a stream of sentences

    :param df: dataframe containing the corpus
    :type df: pandas dataframe
    :param column: dataframe column
    :type column: integer
    :return:
    :rtype:

    :return: texts, which is a list of phrases created
    """
    assert isinstance(df, pd.DataFrame), "A DataFrame has to be passed"

    corpus = list(df[column])
    tokens = [x.split() for x in corpus]
    bigrams = gensim.models.Phrases(tokens)
    texts = [bigrams[line] for line in tokens]
    texts = [" ".join(x) for x in texts]
    return texts


if __name__ == '__main__':
    pass
