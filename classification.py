import networkx as nx
import numpy as np
from collections import OrderedDict
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix
from sklearn.svm import SVC
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
import sys

_score_types = ['micro', 'macro']


def detect_number_of_communities(nxg):
    # It is assumed that the labels of communities starts from 0 to K-1
    max_community_label = -1
    communities = nx.get_node_attributes(nxg, "community")
    # for node in nxg.nodes():
    for node in communities:
        comm_list = communities[node]
        if type(comm_list) is int:
            comm_list = [comm_list]

        c_max = max(comm_list)
        if c_max > max_community_label:
            max_community_label = c_max
    return max_community_label + 1


def read_binary_emb_file(file_path):

    def _int2boolean(num):

        binary_repr = []
        for _ in range(8):

            binary_repr.append(False if num % 2 else True )

            num = num >> 1

        return binary_repr

    with open(file_path, 'rb') as f:
        '''' 
        arr.fromfile(f)

        t = arr.length()
        print(arr)
        print(t)
        '''
        num_of_nodes = int.from_bytes(f.read(4), byteorder='little')
        dim = int.from_bytes(f.read(4), byteorder='little')

        print("{} {}".format(num_of_nodes, dim));
        dimInBytes = int(dim / 8)

        embs = []
        for i in range(num_of_nodes):
            # embs.append(int.from_bytes( f.read(dimInBytes), byteorder='little' ))
            emb = []
            for _ in range(dimInBytes):
                emb.extend(_int2boolean(int.from_bytes(f.read(1), byteorder='little')))
            #print(len(emb))
            embs.append(emb)

    print("=", len(embs[0]))
    return np.asarray(embs, dtype=bool)


def get_node2community(nxg):

    node2community = nx.get_node_attributes(nxg, name='community')

    # for node in nxg.nodes():
    for node in node2community:
        comm = node2community[node]
        if type(comm) == int:
            node2community[node] = [comm]

    return node2community

def evaluate(graph_path, embedding_file, number_of_shuffles, training_ratios, classification_method):

    g = nx.read_gml(graph_path)
    x = read_binary_emb_file(file_path=embedding_file)

    node2community = get_node2community(g)

    # N = g.number_of_nodes()
    K = detect_number_of_communities(g)

    # nodelist = [node for node in g.nodes()]
    nodelist = [int(node) for node in node2community]
    N = len(nodelist)

    #print("--------", x.shape)
    x = x[nodelist, :] #x = np.take(x, nodelist, axis=0)

    label_matrix = [[1 if k in node2community[str(node)] else 0 for k in range(K)] for node in nodelist]
    label_matrix = csr_matrix(label_matrix)


    results = {}

    for score_t in _score_types:
        results[score_t] = OrderedDict()
        for ratio in training_ratios:
            results[score_t].update({ratio: []})

    for train_ratio in training_ratios:

        for _ in range(number_of_shuffles):
            # Shuffle the data
            shuffled_features, shuffled_labels = shuffle(x, label_matrix)

            # Get the training size
            train_size = int(train_ratio * N)
            # Divide the data into the training and test sets
            train_features = shuffled_features[0:train_size, :]
            train_labels = shuffled_labels[0:train_size]

            test_features = shuffled_features[train_size:, :]
            test_labels = shuffled_labels[train_size:]

            # Train the classifier
            if classification_method == "logistic":
                ovr = OneVsRestClassifier(LogisticRegression(solver='liblinear'))
            elif classification_method == "svm-rbf":
                ovr = OneVsRestClassifier(SVC(kernel="rbf", cache_size=4096, probability=True))

            elif classification_method == "svm-hamming":
                ovr = OneVsRestClassifier(SVC(kernel="precomputed", cache_size=4096, probability=True))
                _train_features = train_features.copy()
                _test_features = test_features.copy()

                train_features = 1.0 - cdist(_train_features, _train_features, 'hamming')
                test_features = 1.0 - cdist(_test_features, _train_features, 'hamming')

            elif classification_method == "svm-hamming-cosine":
                ovr = OneVsRestClassifier(SVC(kernel="precomputed", cache_size=4096, probability=True))
                _train_features = train_features.copy()
                _test_features = test_features.copy()

                train_features = 1.0 - cdist(_train_features, _train_features, 'cosine')
                test_features = 1.0 - cdist(_test_features, _train_features, 'cosine')

            else:
                raise ValueError("Invalid classification method name: {}".format(classification_method))

            ovr.fit(train_features, train_labels)

            # Find the predictions, each node can have multiple labels
            test_prob = np.asarray(ovr.predict_proba(test_features))
            y_pred = []
            for i in range(test_labels.shape[0]):
                k = test_labels[i].getnnz()  # The number of labels to be predicted
                pred = test_prob[i, :].argsort()[-k:]
                y_pred.append(pred)

            # Find the true labels
            y_true = [[] for _ in range(test_labels.shape[0])]
            co = test_labels.tocoo()
            for i, j in zip(co.row, co.col):
                y_true[i].append(j)

            mlb = MultiLabelBinarizer(range(K))
            for score_t in _score_types:
                score = f1_score(y_true=mlb.fit_transform(y_true),
                                 y_pred=mlb.fit_transform(y_pred),
                                 average=score_t)

                results[score_t][train_ratio].append(score)


    return results


def get_output_text(results, shuffle_std=False, detailed=False):

    num_of_shuffles = len(list(list(results.values())[0].values())[0])
    train_ratios = [r for r in list(results.values())[0]]
    percentage_title = " ".join("{0:.0f}%".format(100 * r) for r in list(results.values())[0])

    output = ""
    for score_type in _score_types:
        if detailed is True:
            for shuffle_num in range(1, num_of_shuffles + 1):
                output += "{} score, shuffle #{}\n".format(score_type, shuffle_num)
                output += percentage_title + "\n"
                for ratio in train_ratios:
                    output += "{0:.5f} ".format(results[score_type][ratio][shuffle_num - 1])
                output += "\n"

        output += "{} score, mean of {} shuffles\n".format(score_type, num_of_shuffles)
        output += percentage_title + "\n"
        for ratio in train_ratios:
            output += "{0:.5f} ".format(np.mean(results[score_type][ratio]))
        output += "\n"

        if shuffle_std is True:
            output += "{} score, std of {} shuffles\n".format(score_type, num_of_shuffles)
            output += percentage_title + "\n"
            for ratio in train_ratios:
                output += "{0:.5f} ".format(np.std(results[score_type][ratio]))
            output += "\n"

    return output


def print_results(results, shuffle_std, detailed=False):
    output = get_output_text(results=results, shuffle_std=shuffle_std, detailed=detailed)
    print(output)


if __name__ == "__main__":

    graph_path = sys.argv[1]

    embedding_file = sys.argv[2]

    number_of_shuffles = int(sys.argv[3])

    if sys.argv[4] == "large":
        training_ratios = [i for i in np.arange(0.1, 1, 0.1)]
    elif sys.argv[4] == "all":
        training_ratios = [i for i in np.arange(0.01, 0.1, 0.01)] + [i for i in np.arange(0.1, 1, 0.1)]
    else:
        raise ValueError("Invalid training ratio")

    classification_method = sys.argv[5]

    print("---------------------------------------")
    print("Graph path: {}".format(graph_path))
    print("Emb path: {}".format(embedding_file))
    print("Num of shuffles: {}".format(number_of_shuffles))
    print("Training ratios: {}".format(training_ratios))
    print("Classification method: {}".format(classification_method))
    print("---------------------------------------")


    results = evaluate(graph_path, embedding_file, number_of_shuffles, training_ratios, classification_method)

    print_results(results=results, shuffle_std=False, detailed=False)