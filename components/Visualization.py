import networkx as nx
import gensim
import pandas as pd
import community
import matplotlib.pyplot as plt


def network_graph(
    model, plot_title, topics=False, vocab_size=100, word_edges=5, verbose=-1
):
    """
    This function will plot a network using a gensim word2vec model. This will also provide (if required) 
    a list of words belonging to each cluster of network (identified using Louvain Community Detection).
    
    model: pass a word2Vec model generated using Gensim implementation of word2vec.
    plot_title: pass a string with title of the graph you want to print.
    topics: is a boolean parameter (if True will print the list of words in each cluster found using modularity).
    vocab_size: parameter to indicate the number of words you want to use (default = 100).
    word_edges: parameter indicating the minimum number of links a word should have to be included in the network (default = 5)
    """

    G = nx.Graph()
    tupes = []

    for i in list(model.wv.vocab)[:vocab_size]:
        for j in [x[0] for x in model.wv.most_similar(i)[: word_edges + 1]]:
            tupes.append((i, j))

    for i in tupes:
        G.add_edge(i[0], i[1])

    if verbose > 0:
        print(nx.info(G))

    part = community.best_partition(G)
    values = [part.get(node) for node in G.nodes()]
    communities = [(node, part.get(node)) for node in G.nodes()]

    plt.figure(3, figsize=(15, 10))
    plt.title(plot_title, fontsize=20)

    d = dict(G.degree)

    (
        nx.draw_spring(
            G,
            cmap=plt.cm.RdYlBu,
            k=400,
            node_size=[v * 200 for v in d.values()],
            with_labels=True,
            node_color=values,
            edge_color="grey",
        )
    )

    comms = (
        pd.DataFrame(communities, columns=["word", "community"])
        .sort_values(by="community")
        .reset_index()
        .drop(["index"], axis=1)
    )

    if topics == True:
        print("\n")
        for i in range(0, max(comms["community"])):
            print(i, list(comms[comms["community"] == i]["word"]))

    return G, comms

    plt.show()
