from itertools import combinations, groupby
import pandas as pd
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

df = pd.DataFrame(np.random.randint(0,100,size=(15, 4)), columns=list('ABCD'))

def gnp_random_connected_graph(n, p):
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi
    graph, but enforcing that the resulting graph is conneted
    """
    edges = combinations(range(n), 2)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_graph(n, create_using=G)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < p:
                G.add_edge(*e)
    return G

# Plot it
nodes = random.randint(50,100)
seed = random.randint(1,10)
probability = 0.1
G = gnp_random_connected_graph(nodes,probability)

plt.figure(figsize=(10,6))
nx.draw(G, node_color="purple", with_labels=False)
plt.show()