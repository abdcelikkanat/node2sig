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
#include <fstream>

using namespace std;
using namespace Eigen;

void scale(Eigen::SparseMatrix<float, Eigen::RowMajor, ptrdiff_t> &mat);

//void ppmi_matrix(Eigen::MatrixXf &Mat);
template<typename T>
void ppmi_matrix(Eigen::SparseMatrix<T, RowMajor, ptrdiff_t> &Mat);

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
            cout << "+ Error code: " << err_code << endl;
        return 0;
    }

    cout << "------------------------------------" << endl;
    cout << "Walk length: " << walkLen << endl;
    cout << "Dimension: " << dimension << endl;
    cout << "Prob: " << contProb << endl;
    cout << "------------------------------------" << endl;

    auto start_time = chrono::steady_clock::now();

    Eigen::initParallel();

    typedef float T;

    Graph g = Graph(directed);
    g.readEdgeList(edgeFile, verbose);
    //g.writeEdgeList(dataset_path2, true);
    int unsigned numOfNodes = g.getNumOfNodes();
    int unsigned numOfEdges = g.getNumOfEdges();

    // Get edge triplets
    vector <Triplet<T>> edgesTriplets = g.getEdges<T>();
    // Construct the adjacency matrix
    Eigen::SparseMatrix<float, Eigen::RowMajor, ptrdiff_t> A(numOfNodes, numOfNodes);
    A.setFromTriplets(edgesTriplets.begin(), edgesTriplets.end());

    // Normalize the adjacency matrix
    cout << "+ The matrix is being scaled and normalized." << endl;
    scale(A);
    cout << "\t- Completed!" << endl;

    /*
    MatrixXf A1 = MatrixXf(A);
    for(int i=0; i<numOfNodes; i++)
        cout << A1(0, i) << " ";
    cout << endl;
    */
    // Construct zero matrix
    Eigen::SparseMatrix<float, Eigen::RowMajor, ptrdiff_t> X(numOfNodes, numOfNodes);

    // Construct the identity matrix
    Eigen::SparseMatrix<float, Eigen::RowMajor, ptrdiff_t> P(numOfNodes, numOfNodes);
    P.setIdentity();

    // Construct another identity matrix
    Eigen::SparseMatrix<float, Eigen::RowMajor, ptrdiff_t> P0(numOfNodes, numOfNodes);
    P0.setIdentity();


    // Random walk
    if(verbose)
        cout << "+ Random walks are being performed." << endl;
    for(unsigned int l=0; l<walkLen; l++) {
        if(verbose)
            cout << "\t- Current walk: " << l+1 << endl;
        P = P * A;
        P = (contProb)*P + (1-contProb)*P0;
        X = X + P;
    }
    /*
    MatrixXf A2 = MatrixXf(X);
    for(int i=0; i<numOfNodes; i++)
        cout << A2(0, i) << " "; 
    cout << endl;
    */
    if(verbose)
        cout << "\t- Completed!" << endl;

    if(verbose)
        cout << "+ Positive Pointwise Mutual Information (PPMI) is being computed." << endl;
    ppmi_matrix<float>(X);
    if(verbose)
        cout << "\t- Completed!" << endl;
    
    /*
    MatrixXf A3 = MatrixXf(X);
    for(int i=0; i<numOfNodes; i++)
        cout << A3(0, i) << " "; 
    cout << endl;
    */

    /*
    cout << "Writing..." << endl;
    MatrixXf myT = MatrixXf(X); 
    ofstream outfile ("../karate_cpp_pmi.txt");
    for (int r=0; r<numOfNodes; r++) {
        for (int c=0; c< numOfNodes; c++)
        {
        outfile << myT(r,c);
        outfile << " ";
        }
        outfile << "\n";
    }
    outfile.close();
    cout << "-----" << endl;
    */
    Model<T> m(numOfNodes, dimension, verbose);

    if(verbose)
        cout << "+ The hash codes are being generated." << endl;
    if(featureBlockSize == 0 && weightBlockSize ==0)
        m.encodeAllInOne(X, embFile);
    else if(featureBlockSize == 1 && weightBlockSize ==0)
        m.encodeByRow(X, embFile);
    else if(weightBlockSize > 0)
        m.encodeByWeightBlock(X, embFile, weightBlockSize);
    else
        cout << "Invalid settings!" << endl;
    if(verbose)
        cout << "\t- Completed!" << endl;


    auto end_time = chrono::steady_clock::now();
    if(verbose)
        cout << "+ Total elapsed time: " << chrono::duration_cast<chrono::seconds>(end_time - start_time).count() << "secs." << endl;

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

