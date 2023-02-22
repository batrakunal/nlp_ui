"""
This module contains text cleaning functions.
"""
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
for word in ['us', ]:
    stopwords.add(word)


def lower(text):
    """
    Lower-case the text.
    
    :param text: text
    :type text: string
    :return: text in all lower-case characters
    :rtype: string

    .. testsetup::

        from Text_cleaning import lower

    .. doctest::

        >>> text = 'The qUick broWn Fox jumPed oveR tHe laZy Dog.'
        >>> lower(text)
        'the quick brown fox jumped over the lazy dog.'

    """
    return text.lower()

def spacing(text):
    """
    Guarantees a space around punctuation marks.

    :param text: text
    :type text: string
    :return: text with spaces around punctuation marks
    :rtype: string      

    .. testsetup::

        from Text_cleaning import spacing

    .. doctest::

        >>> text = 'The quick brown fox jumped over the lazy dog.'
        >>> spacing(text)
        'The quick brown fox jumped over the lazy dog . '

    """
    text = re.sub(r"\s?([^\w\s'/\-\+$])\s?", r" \1 ", text)
    text = re.sub(r"  ", r" ", text)
    return text

def eliminate_punctuation(text):
    """
    Removes punctuation from text !"#$%&'()*+, -./:;<=>?@[\]^`{|}~

    In particular, when punctuation is inside words without spacing:
        - removes '-' inside words, joining the left and right parts
        - when removing other punctuation characters not adjacent to a whitespace, substitutes the character with a whitespace

    :param text: text
    :type text: string
    :return: text without punctuation characters
    :rtype: string

    .. testsetup::

        from Text_cleaning import eliminate_punctuation

    .. doctest::

        >>> text = 'The# .qu-ick: ~bro-wn% "f-ox, jum-ped{ +ov-er^ .th-e* @la-zy( d-og!'
        >>> eliminate_punctuation(text)
        'The quick brown fox jumped over the lazy dog '
        
        >>> text = 'The# .qu-ick: ~bro-wn% "f-ox,jum-ped{ +ov-er^.th-e* @la-zy( d-og!?'
        >>> eliminate_punctuation(text)
        'The quick brown fox jumped over  the lazy dog  '

        >>> text = 'The .quick$brown"fox jumped{over the@lazy(dog!?'
        >>> eliminate_punctuation(text)
        'The quick brown fox jumped over the lazy dog  '

    """
    inner_word_punctuation = r"-"
    
    punctuation_and_whitespace = r"[^\w\s]+\s+"
    # matches 1 or more punctuation, 1 or more whitespace
    whitespace_and_punctuation = r"\s+[^\w\s]+"
    # matches 1 or more whitespace, 1 or more punctuation

    punctuation = r"[^\w\s]"
    # matches punctuation
    text = re.sub(inner_word_punctuation, '', text)
    text = re.sub(punctuation_and_whitespace, ' ', text)
    text = re.sub(whitespace_and_punctuation, ' ', text)
    text = re.sub(punctuation, ' ', text)
    return text

def eliminate_stopwords(text, stopwords=stopwords):
    """
    Eliminates stopwords from text.

    :param text: text
    :type text: string
    :param stopwords: stopwords
    :type stopwords: list
    :return: text without stopwords
    :rtype: string

    .. testsetup::

        from Text_cleaning import eliminate_stopwords

    .. doctest::

        >>> text = 'The quick brown fox jumped over the lazy dog.'
        >>> eliminate_stopwords(text)
        'quick brown fox jumped lazy dog.'

        >>> text = 'The quick brown fox jumped over the lazy dog.'
        >>> stopwords = ['the']
        >>> eliminate_stopwords(text, stopwords)
        'quick brown fox jumped over lazy dog.'

    """
    return " ".join([x for x in text.split() if x.lower() not in stopwords])

def eliminate_nonalphabetical_characters(text):
    """
    Eliminates non-alphabetical characters from text.

    :param text: text
    :type text: string
    :return: modified text
    :rtype: string

    .. testsetup::

        from Text_cleaning import eliminate_nonalphabetical_characters

    .. doctest::

        >>> text = 'T1h2e3 q4u5i6c7k} b8r9o0w`n~ f!o@x# $j%u^m&p*e(d o)v-e_r= [t+h,e< l>a/z?y; :d"o{g].'
        >>> eliminate_nonalphabetical_characters(text)
        'Thequickbrownfoxjumpedoverthelazydog'
    """
    pattern = '[^a-zA-Z]+'
    text = re.sub(pattern, '', text)
    return text

def text_cleaner(text, lowercase=True, insert_spaces=True, remove_punctuation=True, remove_stopwords=True, stopwords=stopwords):
    """
    Combines the effects of four text cleaning functions.

    .. seealso:: :py:func:`lower`, :py:func:`spacing`, :py:func:`eliminate_punctuation`, :py:func:`eliminate_stopwords`

    :param text: text
    :type text: string
    :param lowercase: need for lowercasing
    :type lowercase: bool
    :param insert_spaces: need for spacing
    :type insert_spaces: bool
    :param remove_punctuation: need for punctuation removal
    :type remove_punctuation: bool
    :param remove_stopwords: need for stopwords removal
    :type remove_stopwords: bool
    :param stopwords: stopwords
    :type stopwords: list
    :return: modified text
    :rtype: string
    """
    if lowercase is True:
        text = lower(text)
    if insert_spaces is True:
        text = spacing(text)
    if remove_punctuation is True:
        text = eliminate_punctuation(text)
    if remove_stopwords is True:
        text = eliminate_stopwords(text, stopwords)
    return text

def chunk_replacement(chunk_list, text):
    """
    Connects words chunks in a text by joining them with an underscore.
    
    :param chunk_list: word chunks
    :type chunk_list: list
    :param text: text
    :type text: string
    :return: text with underscored chunks
    :rtype: string

    .. testsetup::

        from Text_cleaning import chunk_replacement

    .. doctest::

        >>> text = 'The quick brown fox jumped over the lazy dog.'
        >>> chunk_list = ['quick brown fox', 'lazy dog']
        >>> chunk_replacement(chunk_list, text)
        'The quick_brown_fox jumped over the lazy_dog.'

    """
    for chunk in chunk_list:
        text = text.replace(chunk, chunk.replace(' ', '_'))
    return text


if __name__ == '__main__':
    import doctest
    print(doctest.testmod())