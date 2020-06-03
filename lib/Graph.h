#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <limits>
#define EIGEN_USE_MKL_ALL
#include <Eigen/Sparse>
//#include "Matrix.h"

using namespace std;



class Graph {

private:
    bool _directed;
    vector <vector <pair<unsigned int, double>>> _adjList;
    unsigned int _numOfNodes;
    unsigned int _numOfEdges;


    //__unused unsigned int num_of_nodes = 0;
    //__unused unsigned int num_of_edges = 0;
    //__unused vector <vector <int> > edges;
    //__unused vector <vector <int> > adjlist;
    //__unused void vector2Adjlist(bool directed);

public:
    Graph(bool directed);
    Graph();
    ~Graph();

    //vector <int> getCommonNeighbours(int u, int v);
    void readEdgeList(string file_path, bool verbose);
    void writeEdgeList(string file_path, bool weighted);
    unsigned int getNumOfNodes();
    unsigned int getNumOfEdges();
    //template <typename T>
    //inline sparseMatrix<T> getAdjacencyMatrix(bool verbose);

    //__unused void readGraph(string file_path, string filetype, bool directed);
    template <typename T>
    vector<Eigen::Triplet<T>> getEdges();
    //void printAdjList();
    //vector <int> getDegreeSequence();
    vector <vector <pair<unsigned int, double>>> getAdjList();
    //double getClusteringCoefficient(int v, int u);

};

template <typename T>
inline vector<Eigen::Triplet<T>> Graph::getEdges() {

    vector<Eigen::Triplet<T>> edges;

    for (unsigned int node = 0; node < _numOfNodes; node++) {
        for (unsigned int j = 0; j<_adjList[node].size(); j++) {
            edges.push_back(Eigen::Triplet<T>(node, get<0>(_adjList[node][j]), (T)get<1>(_adjList[node][j])));
            // if there is a self-loop, add it once
            if(node != get<0>(_adjList[node][j]))
                edges.push_back(Eigen::Triplet<T>(get<0>(_adjList[node][j]), node, (T)get<1>(_adjList[node][j])));
            //cout << get<1>(_adjList[node][j]) << " ";
        }
    }

    return edges;
}

/*
template <typename T>
inline sparseMatrix<T> Graph::getAdjacencyMatrix(bool verbose) {
    if(verbose)
        cout << "+ The graph is being converted into sparse adjacency matrix format." << endl;

    // Since it is symmetric matrix, it contains twice the number of edges.
    sparseMatrix<T> adjacencyMatrix(_numOfNodes, _numOfNodes, 2*_numOfEdges);

    for(int node=0; node<_numOfNodes; node++) {

        // Since _adjList[node] stores vertices greater than 'node' and the adjacencyMatrix is symmetric
        for(int k = 0; k < node; k++) {
            for (int l = 0; l < _adjList[k].size(); l++) {
                if(_adjList[k][l].first == node)
                    adjacencyMatrix.insert(node, k, _adjList[k][l].second);
            }
        }
        // Add the vertices greater than 'node'
        for(int j=0; j<_adjList[node].size(); j++)
            adjacencyMatrix.insert(node, _adjList[node][j].first, _adjList[node][j].second);
    }

    if(verbose)
        cout << "\t- Completed." << endl;

    return adjacencyMatrix;
}
*/

#endif //GRAPH_H