#import pandas as pd
import networkx as nx
#from matplotlib import pyplot
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
#from Evaluation import PrecisionAtK
#import csv
import numpy as np
#import matplotlib.pyplot as plt
import pickle
from scipy.spatial import distance
from sklearn.utils import shuffle
import sys
import os

# Params
graph_name = "blogcatalog_renaissance"
test_ratio=0.2
max_k = 100
sample_ratio = 0.1
operator = 'hamming' #binary_ops = ['hamming', 'cosine']

#graph_path = "../datasets/{}.gml".format(graph_name)
#graph_path = "/Users/abdulkadir/workspace/NodeSketch/graphs/{}.gml".format(graph_name)
#output_folder="./splitted"

########## --- SPLIT --- ##########

def split_train_test(g, test_ratio):

	# Keep the original graph
	train_g = g.copy()
	train_g.remove_edges_from(nx.selfloop_edges(train_g)) # remove self loops
	test_g = train_g.copy()

	# Split it into two "disjoint" parts
	edges = list(train_g.edges())
	for edge in edges:
		if np.random.rand() < test_ratio:
			train_g.remove_edge(edge[0], edge[1])
		else:
			test_g.remove_edge(edge[0], edge[1])

	# Get the greatest components and relabel
	if not nx.is_connected(train_g):
		print("First size of train graph: {}".format(train_g.number_of_nodes()))
		train_g = train_g.subgraph(max(nx.connected_components(train_g), key=len))
		print("Gcc size of train graph: {}".format(train_g.number_of_nodes()))
		

		train_nodes = list(train_g.nodes())		
		node2newlabel = dict(zip(train_nodes, range(train_g.number_of_nodes())))
		if nx.is_frozen(train_g):
			print("Graph is frozen!")
			train_g = nx.Graph(train_g)
		nx.relabel_nodes(train_g, node2newlabel, copy=False)
		
		test_g = test_g.subgraph(train_nodes)
		if nx.is_frozen(test_g):
			print("Graph is frozen!")
			test_g = nx.Graph(test_g)
		nx.relabel_nodes(test_g, node2newlabel, copy=False)

	return train_g, test_g


def split(graph_path, output_folder ):

	# Read the network
	print("Graph is being read!")
	g = nx.read_gml(graph_path)

	train_g, test_g = split_train_test(g, test_ratio=0.2)

	nx.write_gml(train_g, output_folder+"/"+graph_name+"_gcc_train.gml")
	nx.write_gml(test_g, output_folder+"/"+graph_name+"_gcc_test.gml")

########## --- SPLIT --- ##########

def subraph_sample(g, new_size=None):

	if new_size is None:
		raise ValueError("Enter size for sampling!")

	else:
		sampled_nodes = np.random.choice(g.nodes(), size=new_size, replace=False)

	return g.subgraph(sampled_nodes)


def read_emb_file(file_path, nodelist=None):

	embeddings = {}
	with open(emb_file, 'r') as fin:
		# skip the first line
		fin.readline()
		# read the embeddings

		for line in fin.readlines():
			tokens = line.strip().split()
			if nodelist is None:
				if tokens[0] in nodelist:
					embeddings[tokens[0]] = np.asarray([int(v) for v in tokens[1:]])
			else:
				embeddings[tokens[0]] = np.asarray([int(v) for v in tokens[1:]])

	return embeddings


def get_similarity(embeddings, edgelist=None, metric='hamming'):

	if edgelist is None:

		nodelist = list(embeddings.keys())
		num_of_nodes = len(nodelist)
		scores = []
		for i in range(num_of_nodes):
			for j in range(i+1, num_of_nodes):

				scores.append( ( nodelist[i], nodelist[j], 1.0 - distance.hamming(embeddings[nodelist[i]], embeddings[nodelist[j]]) ) )

	else:

		if metric is 'hamming':
			scores = [( edge[0], edge[1], 1.0 - distance.hamming(embeddings[edge[0]], embeddings[edge[1]]) ) for edge in edgelist]

		else:

			raise ValueError("Invalid hamming!")

	return scores


def computePrecisionCurve(predicted_edge_list, test_g, max_k=-1):

	if max_k == -1:
		max_k = len(predicted_edge_list)
	else:
		max_k = min(max_k, len(predicted_edge_list))

	sorted_edges = sorted(predicted_edge_list, key=lambda x: x[2], reverse=True)

	precision_scores = []
	delta_factors = []
	correct_edge = 0
	for i in range(max_k):
		if test_g.has_edge(sorted_edges[i][0], sorted_edges[i][1]):
			correct_edge += 1
			delta_factors.append(1.0)
		else:
			delta_factors.append(0.0)
		precision_scores.append(1.0 * correct_edge / (i + 1))
	return precision_scores, delta_factors


def computeMAP(edge_score_list, test_g, max_k=-1):

	node_list= list(test_g.nodes())
	number_of_nodes = test_g.number_of_nodes()
	node2triplets = {}
	for node in node_list:
		node2triplets[node]=[]
	for (u, v, weight) in edge_score_list:
		node2triplets[u].append((u, v, weight))
		node2triplets[v].append((v, u, weight))

	node_AP = [0.0] * number_of_nodes
	count = 0
	for i, node in enumerate(node_list):
		count += 1
		precision_scores, delta_factors = computePrecisionCurve(node2triplets[node], test_g, max_k)
		precision_rectified = [p * d for p, d in zip(precision_scores, delta_factors)]
		if sum(delta_factors) == 0:
			node_AP[i] = 0

		else:
			node_AP[i] = float(sum(precision_rectified) / sum(delta_factors))
			
		if len(delta_factors) == 0:
			count -= 1

	assert count > 0, "Count must be greater than zero!"

	return sum(node_AP) / count


def get_map_score(input_folder, emb_file, max_k, sample_ratio=None):

	#g = nx.read_gml(input_folder + "/" + graph_name + "_gcc.gml")
	#num_of_nodes = g.number_of_nodes()

	train_g = nx.read_gml(input_folder + "/" + graph_name + "_gcc_train.gml")
	test_g = nx.read_gml(input_folder + "/" + graph_name + "_gcc_test.gml")

	
	print(train_g.number_of_nodes(), train_g.number_of_edges() ) 
	print(test_g.number_of_nodes(), test_g.number_of_edges() ) 

	print("train and test sets are read!")
	
	new_size = None #test_g.number_of_nodes()
	
	#test_g = subraph_sample(g=test_g, new_size=new_size)
	#print("Sub sampling ok!")
	
	embeddings = read_emb_file(file_path=emb_file, nodelist=list(test_g.nodes()))
	print("embeddings ok!")

	scores_edges = []
	if sample_ratio is None:

		test_g_nodes = list(test_g.nodes())
		for i in range(test_g.number_of_nodes()):
			for j in range(i+1, test_g.number_of_nodes()):
				#for edge in test_g.edges():
				edge = (test_g_nodes[i], test_g_nodes[j])
				if (edge[0] != edge[1]) and (train_g.has_edge(edge[0], edge[1]) is False):
					scores_edges.append(edge)

	else:
		print("Sample Ratio", sample_ratio)

		test_g_nodes = list(test_g.nodes())
		num_of_nodes = len(test_g_nodes)
		sample_size = int( sample_ratio * num_of_nodes * (num_of_nodes-1) / 2)

		counter = 0
		sampled_idx_list = [[] for _ in range(num_of_nodes)]
		while counter < sample_size:
			candidate = np.random.randint(num_of_nodes, size=2)
			if candidate[0] != candidate[1] and candidate[0] not in sampled_idx_list[candidate[1]]:

				sampled_idx_list[candidate[0]].append(candidate[1])
				sampled_idx_list[candidate[1]].append(candidate[0])

				u = test_g_nodes[candidate[0]]
				v = test_g_nodes[candidate[1]]
				
				if train_g.has_edge(u, v) is False:
					scores_edges.append( (u, v) )

					counter += 1

	scores = get_similarity(embeddings, edgelist=scores_edges)
	print("similarity ok!")

	result = computeMAP(edge_score_list=scores, test_g=test_g, max_k=max_k)
	print("MAP ok!")

	
	return result


if sys.argv[1] == 'split':
	graph_path = sys.argv[2]
	output_folder = sys.argv[3]

	print("Graph path: {}".format(graph_path))
	print("Output folder path: {}".format(output_folder))

	#split(graph_path, output_folder)
	g = nx.read_gml(graph_path)
	print(g.number_of_nodes())

elif sys.argv[1] == 'predict':

	emb_file = sys.argv[2]
	input_folder = sys.argv[3]

	sample_ratio = 0.1 #new_size = 1024
	max_k = 100

	print("Emb path: {}".format(emb_file))
	print("Output folder path: {}".format(input_folder))

	map_scores = []
	for repeat in range(5):
		map_score = get_map_score(input_folder, emb_file, max_k=max_k, sample_ratio=sample_ratio,)
		print(map_score)
		map_scores.append(map_score)

	map_avg = np.mean(map_scores)

	print("Avg: {}".format(map_avg))

else:

	raise ValueError("Enter split or predict")
