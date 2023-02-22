

if __name__ == '__main__':
    from Text_cleaning import text_cleaner
    from Chunking import chunking, chunking_count, np_chunking, chunking_NER, chunking_soa, chunking_fair
    import sample_text

    from metrics import entropy

    print('start')

    partial = 10

    text = sample_text.text_whole
    text = text[:int(len(text)/partial)]
    n_partitions = int(100/partial)
    text_list = [text[int(x * len(text) / n_partitions) : int((x + 1) * len(text) / n_partitions)] for x in range(n_partitions)]

    for text in text_list[:1]:

        text = text_cleaner(text)
        # print(text)

        chunk_list = np_chunking(text)
        counts = chunking_count(chunk_list)
        print(counts.most_common(5))

        chunk_list = chunking_NER(text)
        counts = chunking_count(chunk_list)
        print(counts.most_common(5))

        chunk_list = chunking_soa(text)
        counts = chunking_count(chunk_list)
        print(counts.most_common(5))

        chunk_list = chunking_fair(text)
        counts = chunking_count(chunk_list)
        print(counts.most_common(5))

        # chunk_list = chunking(text)
        # counts = chunking_count(chunk_list)
        # print(counts.most_common(5))
        # print(entropy(chunk_list))





