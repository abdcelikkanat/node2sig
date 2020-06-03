#define EIGEN_USE_MKL_ALL
#include "mkl.h"
#include <Eigen/Sparse>
#include <vector>
#include <iostream>
#include <string>
#include <math.h>
#include "Graph.h"
#include "Model.h"
#include "Utilities.h"

using namespace std;
using namespace Eigen;

void scale(Eigen::SparseMatrix<float, Eigen::RowMajor> &mat);

//void ppmi_matrix(Eigen::MatrixXf &Mat);
template<typename T>
void ppmi_matrix(Eigen::SparseMatrix<T, RowMajor> &Mat);

int main(int argc, char** argv) {

    //string dataset_path = "/Users/abdulkadir/workspace/nodesig/cplusplus/tests/karate.edgelist";
    //string dataset_path = "/Users/abdulkadir/workspace/nodesig/cplusplus/Homo_sapiens_renaissance.edgelist";
    //string embFilePath = "/Users/abdulkadir/workspace/nodesig/cplusplus/deneme.embedding";

    //string dataset_path = "/home/abdulkadir/Desktop/datasets/Homo_sapiens_renaissance.edgelist";
    //string embFilePath = "/home/abdulkadir/Desktop/nodesig/cplusplus/Homo_sapiens_renaissance.embedding";
    //string dataset_path = "/home/abdulkadir/Desktop/datasets/youtube_renaissance.edgelist";
    //string embFilePath = "/home/abdulkadir/Desktop/nodesig/cplusplus/youtube_renaissance.embedding";

    //string dataset_path = "/home/kadir/workspace/cplusplus/youtube_renaissance.edgelist";
    //string embFilePath = "/home/kadir/workspace/cplusplus/youtube_renaissance.embedding";


    string edgeFile, embFile;
    unsigned int walkLen = 0;

    // Default values
    bool directed = false;
    unsigned int dimension = 8192;
    float contProb = 0.98;
    int featureBlockSize = 0;
    int weightBlockSize = 0;
    bool verbose = true;


    int err_code =  parse_arguments(argc, argv, edgeFile, embFile, walkLen,
            dimension, contProb, featureBlockSize, weightBlockSize, verbose);


    if(err_code != 0) {
        if(err_code < 0)
            cout << "Error code: " << err_code << endl;
        return 0;
    }

    typedef float T;

    Graph g = Graph(directed);
    g.readEdgeList(edgeFile, verbose);
    //g.writeEdgeList(dataset_path2, true);
    int unsigned numOfNodes = g.getNumOfNodes();
    int unsigned numOfEdges = g.getNumOfEdges();

    // Get edge triplets
    vector <Triplet<T>> edgesTriplets = g.getEdges<T>();
    // Construct the adjacency matrix
    Eigen::SparseMatrix<float, Eigen::RowMajor> A(numOfNodes, numOfNodes);
    A.setFromTriplets(edgesTriplets.begin(), edgesTriplets.end());

    // Normalize the adjacency matrix
    cout << "Scaling started!" << endl;
    //scale(A);
    cout << "Scaling completed!" << endl;

    // Construct zero matrix
    Eigen::SparseMatrix<float, Eigen::RowMajor> X(numOfNodes, numOfNodes);

    // Construct the identity matrix
    Eigen::SparseMatrix<float, Eigen::RowMajor> P(numOfNodes, numOfNodes);
    P.setIdentity();

    // Construct another identity matrix
    Eigen::SparseMatrix<float, Eigen::RowMajor> P0(numOfNodes, numOfNodes);
    P0.setIdentity();


    // Random walk
    if(verbose)
        cout << "Performing random walks" << endl;
    for(unsigned int l=0; l<walkLen; l++) {
        if(verbose)
            cout << "--> Current walk: " << l+1 << endl;
        P = P * A;
        P = (contProb)*P + (1-contProb)*P0;
        X = X + P;
    }
    if(verbose)
        cout << "Completed!" << endl;

    if(verbose)
        cout << "Computing Positive Pointwise Mutual Information (PPMI)" << endl;
    ppmi_matrix<float>(X);
    if(verbose)
        cout << "Completed!" << endl;

    Model<T> m(numOfNodes, dimension, verbose);
    cout << "Model started!" << endl;

    if(verbose)
        cout << "The hash codes are being generated." << endl;
    if(featureBlockSize == 0 && weightBlockSize ==0)
        m.encodeAllInOne(X, embFile);
    else if(featureBlockSize == 1 && weightBlockSize ==0)
        m.encodeByRow(X, embFile);
    else if(weightBlockSize > 0)
        m.encodeByWeightBlock(X, embFile, weightBlockSize);
    else
        cout << "Invalid settings!" << endl;
    if(verbose)
        cout << "Completed!" << endl;


    return 0;

}


void scale(Eigen::SparseMatrix<float, Eigen::RowMajor> &mat) {

    float minValue, maxValue;

    auto values = mat.coeffs();
    //for(unsigned int i=0; i<values.size(); i++)
    //    cout << values[i] << endl;

    // Set diagonals
    auto diagonals = mat.diagonal();
    for(int d=0; d<diagonals.size(); d++) {
        diagonals.coeffRef(d, d) = 0;
        if(mat.row(d).sum() == 0)
            diagonals.coeffRef(d,d)=1;
    }

    // Normalize row sums
    float rowSum;
    for(int i=0; i<mat.outerSize(); i++) {
        rowSum = 0;
        for(Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(mat, i); it; ++it)
            rowSum += it.value();

        if(rowSum != 0) {
            for(Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(mat, i); it; ++it)
                it.valueRef() = it.valueRef() / rowSum;
        }
    }


}

template<typename T>
void ppmi_matrix(Eigen::SparseMatrix<T, RowMajor> &Mat) {

    // Positive Pointwise Mutual Information matrix
    T *rowSums = new T[Mat.rows()];
    T *colSums = new T[Mat.cols()];
    T totalSum = 0;

    for (int row=0; row < Mat.rows(); ++row) {
        for (typename SparseMatrix<T, RowMajor>::InnerIterator it(Mat, row); it; ++it) {
            rowSums[row] += it.value();
            colSums[it.col()] += it.value();
            totalSum += it.value();
        }
    }

    T value;
    vector <Triplet<T>> valueTriplets;
    for (unsigned int row=0; row < Mat.rows(); ++row) {
        for (typename SparseMatrix<T, RowMajor>::InnerIterator it(Mat, row); it; ++it) {
            value = log( (totalSum*it.value()) / (rowSums[row]*colSums[it.col()]) );
            if(value > 0)
                valueTriplets.push_back(Triplet<T>(row, it.col(), totalSum));
        }
    }

    delete [] rowSums;
    delete [] colSums;

    Eigen::SparseMatrix<T, RowMajor> tempMat(Mat.rows(), Mat.cols());
    tempMat.setFromTriplets(valueTriplets.begin(), valueTriplets.end());


    Mat = tempMat;


}

