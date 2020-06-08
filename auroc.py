import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import sys
import os
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, model_selection, pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn import preprocessing
import pickle
from sklearn.metrics import precision_recall_curve

from scipy.spatial import distance


def split_into_training_test_sets(g, test_set_ratio):


    print("+ Getting the gcc of the original graph.")
    # Keep the original graph
    train_g = g.copy()
    train_g.remove_edges_from(nx.selfloop_edges(train_g)) # remove self loops
    train_g = train_g.subgraph(max(nx.connected_components(train_g), key=len))
    if nx.is_frozen(train_g):
        train_g = nx.Graph(train_g)
    print("\t- Completed!")

    print("+ Relabeling.")
    node2newlabel = {node: str(nodeIdx) for nodeIdx, node in enumerate(train_g.nodes())}
    train_g = nx.relabel_nodes(G=train_g, mapping=node2newlabel, copy=True)
    print("\t- Completed!")

    num_of_nodes = train_g.number_of_nodes()
    nodelist = list(train_g.nodes())

    print("+ Splitting into train and test sets.")
    num_of_edges = train_g.number_of_edges()

    edges = list(train_g.edges())

    test_size = int(test_set_ratio * num_of_edges)

    test_g = nx.Graph()
    test_g.add_nodes_from(train_g.nodes())

    count = 0
    idx = 0
    perm = np.arange(len(edges))
    while(count < test_size and count < num_of_edges):
        if count % 10000 == 0:
            print("{}/{}".format(count, test_size))
        # Remove the chosen edge
        chosen_edge = edges[perm[idx]]
        train_g.remove_edge(chosen_edge[0], chosen_edge[1])
        if chosen_edge[1] in nx.connected._plain_bfs(train_g, chosen_edge[0]):
            test_g.add_edge(chosen_edge[0], chosen_edge[1])
            count += 1
        else:
            train_g.add_edge(chosen_edge[0], chosen_edge[1])

        idx += 1
    print("--> Completed!")

    if count != test_size:
        raise ValueError("Enough positive edge samples could not be found!")


    # Generate the negative samples
    print("\+ Generating negative samples")
    count = 0
    negative_samples_idx = [[] for _ in range(num_of_nodes)]
    negative_samples = []
    while count < 2*test_size:
        if count % 10000 == 0:
            print("{}/{}".format(count, 2*test_size))
        uIdx = np.random.randint(num_of_nodes-1)
        vIdx = np.random.randint(uIdx+1, num_of_nodes)

        if vIdx not in negative_samples_idx[uIdx]:
            negative_samples_idx[uIdx].append(vIdx)

            u = nodelist[uIdx]
            v = nodelist[vIdx]

            negative_samples.append((u,v))

            count += 1

    train_neg_samples = negative_samples[:test_size]
    test_neg_samples = negative_samples[test_size:test_size*2]

    return train_g, test_g, train_neg_samples, test_neg_samples

############################################################################################################


def read_emb_file(file_path, file_type="binary"):

    if file_type == "binary":

        def _int2boolean(num):
            binary_repr = []
            for _ in range(8):
                binary_repr.append(False if num % 2 else True )
                num = num >> 1
            return binary_repr

        with open(file_path, 'rb') as f:

            num_of_nodes = int.from_bytes(f.read(4), byteorder='little')
            dim = int.from_bytes(f.read(4), byteorder='little')

            print("{} {}".format(num_of_nodes, dim));
            dimInBytes = int(dim / 8)

            embs = []
            for i in range(num_of_nodes):
                emb = []
                for _ in range(dimInBytes):
                    emb.extend(_int2boolean(int.from_bytes(f.read(1), byteorder='little')))
                embs.append(emb)

            embs = np.asarray(embs, dtype=bool)

    else:

        with open(file_path, 'r') as fin:
            # Read the first line
            num_of_nodes, dim = ( int(token) for token in fin.readline().strip().split() )

            # read the embeddings
            embs = [[] for _ in range(num_of_nodes)]

            for line in fin.readlines():
                tokens = line.strip().split()
                embs[int(tokens[0])] = [float(v) for v in tokens[1:]]

            embs = np.asarray(embs, dtype=np.float)

    return embs


def extract_feature_vectors_from_embeddings(edges, embeddings, binary_operator):

    features = []
    for i in range(len(edges)):
        edge = edges[i]
        vec1 = embeddings[int(edge[0])]
        vec2 = embeddings[int(edge[1])]

        if binary_operator == "hamming":
            value = 1.0 - distance.hamming(vec1, vec2)
        elif binary_operator == "cosine":
            value = 1.0 - distance.cosine(vec1, vec2)
        else:
            raise ValueError("Invalid operator!")

        features.append(value)


    features = np.asarray(features)
    # Reshape the feature vector if it is 1d vector
    if binary_operator in ["hamming", "cosine"]:
        features = features.reshape(-1, 1)

    return features

################################################################################################

def split(graph_path, output_folder ):

    # Read the network
    print("Graph is being read!")
    g = nx.read_gml(graph_path)

    train_g, test_g, train_neg_samples, test_neg_samples = split_into_training_test_sets(g, test_set_ratio=0.2)

    print("Train ratio: {}".format(train_g.number_of_edges()/float(g.number_of_edges())))
    print("Test ratio: {}".format(test_g.number_of_edges()/float(g.number_of_edges())))

    nx.write_gml(train_g, output_folder+"/"+os.path.splitext(os.path.basename(graph_path))[0]+"_gcc_train.gml")
    nx.write_edgelist(train_g, output_folder+"/"+os.path.splitext(os.path.basename(graph_path))[0]+"_gcc_train.edgelist", data=['weight'])
    nx.write_gml(test_g, output_folder+"/"+os.path.splitext(os.path.basename(graph_path))[0]+"_gcc_test.gml")

    np.save(output_folder+"/"+os.path.splitext(os.path.basename(graph_path))[0]+"_gcc_train_negative_samples.npy", train_neg_samples)
    np.save(output_folder+"/"+os.path.splitext(os.path.basename(graph_path))[0]+"_gcc_test_negative_samples.npy", test_neg_samples)

def predict(input_folder, graph_name, emb_file, file_type, binary_operator):

    print("-----------------------------------------------")
    print("Input folder: {}".format(input_folder))
    print("Graph name: {}".format(graph_name))
    print("Emb path: {}".format(emb_file))
    print("File type: {}".format(file_type))
    print("Metric type: {}".format(binary_operator))
    print("-----------------------------------------------")

    test_g = nx.read_gml(input_folder+"/"+graph_name+"_gcc_test.gml")
    test_neg_samples = np.load(input_folder+"/"+graph_name+"_gcc_test_negative_samples.npy")

    test_samples = [list(edge) for edge in test_g.edges()] + test_neg_samples.tolist()
    test_labels = [1 for _ in test_g.edges()] + [0 for _ in test_neg_samples]

    embs = read_emb_file(emb_file, file_type=file_type)

    test_features = extract_feature_vectors_from_embeddings(edges=test_samples,
                                                            embeddings=embs,
                                                            binary_operator=binary_operator)

    clf = LogisticRegression()
    clf.fit(test_features, test_labels)

    test_preds = clf.predict_proba(test_features)[:, 1]

    test_roc = roc_auc_score(y_true=test_labels, y_score=test_preds)

    print("Roc auc score: {}".format(test_roc))

    precision, recall, _ = precision_recall_curve(test_labels, test_preds)

    np.save(input_folder+"/"+graph_name+"_gcc_precision_recall.npy", [precision, recall])


if sys.argv[1] == 'split':

    graph_path = sys.argv[2]
    output_folder = sys.argv[3]
    split(graph_path, output_folder)

elif sys.argv[1] == 'predict':

    input_folder = sys.argv[2]
    graph_name = sys.argv[3]
    emb_file = sys.argv[4]
    file_type = sys.argv[5] #binary
    binary_operator = sys.argv[6] #hamming

    predict(input_folder, graph_name, emb_file, file_type, binary_operator)

