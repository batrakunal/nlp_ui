import scipy.stats
from Chunking import chunking_count

def entropy(chunk_list):
    """
    Evaluate the entropy of a distribution of chunks.

    :param chunk_list: chunks
    :type chunk_list: list
    :return: entropy of the chunks
    :rtype: float


    """
    freq_list = list(chunking_count(chunk_list).values())
    # freq_sum = sum(freq_list)
    # freq_list = [x/freq_sum for x in freq_list]
    return scipy.stats.entropy(freq_list, base=2)

