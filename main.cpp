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

void scale(Eigen::SparseMatrix<float, Eigen::RowMajor, ptrdiff_t> &mat);

//void ppmi_matrix(Eigen::MatrixXf &Mat);
template<typename T>
void ppmi_matrix(Eigen::SparseMatrix<T, RowMajor, ptrdiff_t> &Mat);

int main(int argc, char** argv) {


    string edgeFile, embFile;
    unsigned int walkLen = 0;

    // Default values
    bool directed = false;
    unsigned int dimension = 8192;
    float contProb = 0.98;
    bool verbose = true;


    int err_code =  parse_arguments(argc, argv, edgeFile, embFile, walkLen, dimension, contProb, verbose);


    if(err_code != 0) {
        if(err_code < 0)
            cout << "+ Error code: " << err_code << endl;
        return 0;
    }

    cout << "------------------------------------" << endl;
    cout << "Walk length: " << walkLen << endl;
    cout << "Dimension: " << dimension << endl;
    cout << "Prob: " << contProb << endl;
    cout << "------------------------------------" << endl;

    auto start_time = chrono::steady_clock::now();


    typedef float T;

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
        for(unsigned int nbIdx=0; nbIdx<adjList[node].size(); nbIdx++) {
            pair <unsigned int, double> p = adjList[node][nbIdx];
            P[node].push_back(p);
            if(get<0>(p) != node)
                P[get<0>(p)].push_back(pair <unsigned int, double>(node, get<1>(p)));
        }
    }


    T rowSum;
    for(unsigned int node=0; node<numOfNodes; node++) {
        rowSum = 0.0;
        for(unsigned int nbIdx=0; nbIdx<P[node].size(); nbIdx++)
            rowSum += get<1>(P[node][nbIdx]);
        for(unsigned int nbIdx=0; nbIdx<P[node].size(); nbIdx++)
            get<1>(P[node][nbIdx]) = get<1>(P[node][nbIdx]) / rowSum;
    }

    Model<T> m(numOfNodes, dimension, verbose);
    m.learnEmb(P, walkLen, embFile);

    auto end_time = chrono::steady_clock::now();
    if (verbose)
        cout << "+ Total elapsed time: " << chrono::duration_cast<chrono::seconds>(end_time - start_time).count()
             << "secs." << endl;

    return 0;

}


void scale(Eigen::SparseMatrix<float, Eigen::RowMajor, ptrdiff_t> &mat) {

    auto maxValue = mat.coeffs().maxCoeff();
    mat = mat / maxValue;

    // Set diagonals
    /*
    auto diagonals = mat.diagonal();
    for(int d=0; d<diagonals.size(); d++) {
        diagonals.coeffRef(d, d) = 0;
        if(mat.row(d).sum() == 0)
            diagonals.coeffRef(d,d)=1;
    }
    */

    // Normalize row sums
    float rowSum;
    for(int i=0; i<mat.outerSize(); i++) {
        rowSum = 0;
        for(Eigen::SparseMatrix<float, Eigen::RowMajor, ptrdiff_t>::InnerIterator it(mat, i); it; ++it)
            rowSum += it.value();

        if(rowSum != 0) {
            for(Eigen::SparseMatrix<float, Eigen::RowMajor, ptrdiff_t>::InnerIterator it(mat, i); it; ++it)
                it.valueRef() = it.valueRef() / rowSum;
        }
    }


}

template<typename T>
void ppmi_matrix(Eigen::SparseMatrix<T, RowMajor, ptrdiff_t> &Mat) {

    // Positive Pointwise Mutual Information matrix
    T *rowSums = new T[Mat.rows()];
    T *colSums = new T[Mat.cols()];
    T totalSum = 0;

    for (int row=0; row < Mat.rows(); row++)
        rowSums[row] = 0.0;

    for (int col=0; col < Mat.cols(); col++)
        colSums[col] = 0.0;


    scale(Mat);

    for (int row=0; row < Mat.rows(); ++row) {
        for (typename SparseMatrix<T, RowMajor, ptrdiff_t>::InnerIterator it(Mat, row); it; ++it) {
            if(it.col() == row) {
               it.valueRef() = 0;
            } else {
               rowSums[row] += it.value();
               colSums[it.col()] += it.value();
               totalSum += it.value();
            }
        }
        /*
        for (typename SparseMatrix<T, RowMajor, ptrdiff_t>::InnerIterator it(Mat, row); it; ++it) {
            if(row == 0 && it.col() ==0)
                cout << "YETERRRR: " << it.value() << endl;

            if(it.col() == row)
                it.valueRef() = 0;
            else
                it.valueRef() = it.value() / rowSums[row];
            colSums[it.col()] += it.value();
            totalSum += it.value();
        }
        rowSums[row] = 1.0;
        */
    }
    cout << "Totsl sum: " << totalSum << endl;

    T value;
    vector <Triplet<T>> valueTriplets;
    for (unsigned int row=0; row < Mat.rows(); ++row) {
        for (typename SparseMatrix<T, RowMajor, ptrdiff_t>::InnerIterator it(Mat, row); it; ++it) {

            if(it.value() > 0) {
                value = log( (totalSum*it.value()) / (rowSums[row]*colSums[it.col()]) );
                if(value > 0) {
                    valueTriplets.push_back(Triplet<T>(row, it.col(), value));
                    //cout << value << " " << endl;
                }
            }
        }
    }

    delete [] rowSums;
    delete [] colSums;

    Eigen::SparseMatrix<T, RowMajor, ptrdiff_t> tempMat(Mat.rows(), Mat.cols());
    tempMat.setFromTriplets(valueTriplets.begin(), valueTriplets.end());


    Mat = tempMat;


}

