#include <vector>
#include <iostream>
#include <string>
#include <math.h>
#include "Graph.h"
#include "Model.h"
#include "Utilities.h"
#include <fstream>

using namespace std;
using namespace Eigen;


template<typename T>
void normalizeRows(vector <vector <pair<unsigned int, T>>> &P) {

    T rowSum;
    for(unsigned int node=0; node<P.size(); node++) {
        rowSum = 0.0;
        for(unsigned int nbIdx=0; nbIdx<P[node].size(); nbIdx++)
            rowSum += get<1>(P[node][nbIdx]);
        for(unsigned int nbIdx=0; nbIdx<P[node].size(); nbIdx++)
            get<1>(P[node][nbIdx]) = get<1>(P[node][nbIdx]) / rowSum;
    }

}

int main(int argc, char** argv) {

    typedef float T;
    string edgeFile, embFile;
    unsigned int walkLen = 0;

    // Default values
    bool directed = false;
    unsigned int dimension = 8192;
    T alpha = 1.0;
    bool verbose = true;


    int err_code =  parse_arguments(argc, argv, edgeFile, embFile, walkLen, dimension, alpha, verbose);


    if(err_code != 0) {
        if(err_code < 0)
            cout << "+ Error code: " << err_code << endl;
        return 0;
    }

    cout << "------------------------------------" << endl;
    cout << "Walk length: " << walkLen << endl;
    cout << "Dimension: " << dimension << endl;
    cout << "Alpha: " << alpha << endl;
    cout << "------------------------------------" << endl;

    auto start_time = chrono::steady_clock::now();


    Graph g = Graph(directed);
    g.readEdgeList(edgeFile, verbose);
    //g.writeEdgeList(dataset_path2, true);
    int unsigned numOfNodes = g.getNumOfNodes();
    int unsigned numOfEdges = g.getNumOfEdges();

    // Get edge triplets
    vector <vector <pair<unsigned int, double>>> adjList = g.getAdjList();
    vector <vector <pair<unsigned int, T>>> P;
    P.resize(numOfNodes);

    for(unsigned int node=0; node<numOfNodes; node++) {

        P[node].push_back(pair<unsigned int, double>(node, 1.0));

        for(unsigned int nbIdx=0; nbIdx<adjList[node].size(); nbIdx++) {
            pair <unsigned int, double> p = adjList[node][nbIdx];
            if(get<0>(p) > node) {
                P[node].push_back(p);
                P[get<0>(p)].push_back(pair<unsigned int, double>(node, get<1>(p)));
            }
        }
    }


    normalizeRows(P);


    Model<T> m(numOfNodes, dimension, verbose);
    m.learnEmb(P, walkLen, alpha, embFile);

    auto end_time = chrono::steady_clock::now();
    if (verbose)
        cout << "+ Total elapsed time: " << chrono::duration_cast<chrono::seconds>(end_time - start_time).count()
             << "secs." << endl;

    return 0;

}

