#ifndef MODEL_H
#define MODEL_H

#include <iostream>
#include <random>
#include <sstream>
#include <fstream>
#include <string>
#include "Graph.h"
#include "mkl.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <bitset>
#include <chrono>

using namespace std;

double sample_from_normal(float var)
{
    static mt19937 rng;
    static normal_distribution<float> nd(0.0,1.0);
    return nd(rng);
}


template<typename T>
inline bool sign_func(T x) {
    if (x > 0)
        return true;
    else
        return false;
}


template<typename T>
class Model {
private:
    unsigned int _dim;
    unsigned int _numOfNodes;
    Eigen::MatrixXf _weights;
    bool _verbose;
    int _headerBlockSize = 4; // in bytes
    random_device _rd;
    void _generateWeights(unsigned int N, unsigned int M);

public:

    Model(unsigned int numOfNodes, unsigned int dim, bool verbose);
    ~Model();
    void encodeByRow(Eigen::SparseMatrix<T, Eigen::RowMajor> &X, string filePath);
    void encodeAllInOne(Eigen::SparseMatrix<T, Eigen::RowMajor> &X, string filePath);
    void encodeByWeightBlock(Eigen::SparseMatrix<T, Eigen::RowMajor> &X, string filePath, int weightBlockSize);



};

template<typename T>
void Model<T>::_generateWeights(unsigned int N, unsigned int M) {

    if (this->_verbose)
        cout << "A weight matrix of size " << this->_numOfNodes << "x" << this->_dim << " is being (re)generated." << endl;

    default_random_engine generator(this->_rd());
    normal_distribution<T> distribution(0.0, 1.0);
    this->_weights = Eigen::MatrixXf::Zero(N, M);
    this->_weights = this->_weights.unaryExpr([&](float dummy) { return distribution(generator); });

    if (this->_verbose)
        cout << "--> Completed!" << endl;

}

template<typename T>
Model<T>::Model(unsigned int numOfNodes, unsigned int dim, bool verbose) {

    if(dim % 8 != 0) {
        cout << "The embedding dimension must be divisible by 8." << endl;
        throw;
    }

    this->_dim = dim;
    this->_numOfNodes = numOfNodes;
    this->_verbose = verbose;

}

template<typename T>
Model<T>::~Model() {

}


template<typename T>
void Model<T>::encodeByRow(Eigen::SparseMatrix<T, Eigen::RowMajor> &X, string filePath) {

    this->_generateWeights(this->_numOfNodes, this->_dim);

    fstream fs(filePath, fstream::out | fstream::binary);
    if(fs.is_open()) {

        // Write the header
        fs.write(reinterpret_cast<const char *>(&this->_numOfNodes), this->_headerBlockSize);
        fs.write(reinterpret_cast<const char *>(&this->_dim), this->_headerBlockSize);

        // Get the initial time
        auto start_time = chrono::steady_clock::now();

        for(int currentRowIdx=0; currentRowIdx < this->_numOfNodes; currentRowIdx++) {

            if(currentRowIdx % 10000 == 0 && this->_verbose)
                cout << "Current row: " << currentRowIdx + 1 << "/" << this->_numOfNodes << endl;

            Eigen::VectorXf rowProd = X.row(currentRowIdx) * this->_weights;
            vector<uint8_t> bin(this->_dim/8, 0);
            for (unsigned int d = 0; d < this->_dim; d++) {
                bin[int(d/8)] <<= 1;
                if (rowProd.coeff(d) > 0)
                    bin[int(d/8)] += 1;
            }

            // Write to the file
            copy(bin.begin(), bin.end(), std::ostreambuf_iterator<char>(fs));

        }

        auto end_time = chrono::steady_clock::now();
        if(_verbose)
            cout << "--> Elapsed time for matrix multiplication: " << chrono::duration_cast<chrono::seconds>(end_time - start_time).count() << endl;

        fs.close();

    } else {
        cout << "An error occurred during opening the file!" << endl;
    }

}

template<typename T>
void Model<T>::encodeAllInOne(Eigen::SparseMatrix<T, Eigen::RowMajor> &X, string filePath) {

    _generateWeights(this->_numOfNodes, this->_dim);

    fstream fs(filePath, fstream::out | fstream::binary);

    if(fs.is_open()) {

        Eigen::MatrixXf matrixProd(_numOfNodes, _dim);

        // Write the header
        fs.write(reinterpret_cast<const char *>(&_numOfNodes), _headerBlockSize);
        fs.write(reinterpret_cast<const char *>(&_dim), _headerBlockSize);

        // Get the current time
        auto start_time = chrono::steady_clock::now();

        // Compute the matrix multiplication
        matrixProd = X * _weights;

        auto end_time = chrono::steady_clock::now();
        if(_verbose)
            cout << "--> Elapsed time for matrix multiplication: "<< chrono::duration_cast<chrono::seconds>(end_time - start_time).count() << endl;


        for(unsigned int currentRowIdx=0; currentRowIdx< this->_numOfNodes; currentRowIdx++) {

            Eigen::VectorXf nodeVect = matrixProd.row(currentRowIdx);

            vector<uint8_t> bin(_dim/8, 0);
            for (unsigned int d = 0; d < _dim; d++) {
                bin[int(d/8)] <<= 1;
                if (nodeVect.coeff(d) > 0)
                    bin[int(d/8)] += 1;
            }

            copy(bin.begin(), bin.end(), std::ostreambuf_iterator<char>(fs));

        }

        fs.close();

    } else {
        cout << "An error occurred during opening the file!" << endl;
    }

}


template<typename T>
void Model<T>::encodeByWeightBlock(Eigen::SparseMatrix<T, Eigen::RowMajor> &X, string filePath, int weightBlockSize) {

    fstream fs(filePath, fstream::out | fstream::binary);

    bool **embMatrix = new bool*[this->_numOfNodes];
    for(unsigned int node=0; node < this->_numOfNodes; node++)
        embMatrix[node] = new bool[this->_dim];

    // Get the transpose of feature matrix
    auto Xt = X.transpose();

    unsigned int initialBlockIdx, lastBlockIdx;
    unsigned int currentBlockSize;

    initialBlockIdx = 0;
    while(initialBlockIdx < this->_dim) {

        // Get the current weight block size
        lastBlockIdx = initialBlockIdx + weightBlockSize;
        if( lastBlockIdx > this->_dim)
            lastBlockIdx = this->_dim;
        currentBlockSize = lastBlockIdx - initialBlockIdx;

        // (Re)Generate weights
        this->_generateWeights(currentBlockSize, this->_numOfNodes);

        // Compute the matrix multiplication
        Eigen::MatrixXf blockProd = this->_weights * Xt;

        for(unsigned int w=initialBlockIdx; w < lastBlockIdx; w++ ) {
            for(unsigned int n=0; n<this->_numOfNodes; n++)
                embMatrix[n][w] = blockProd.coeff(w-initialBlockIdx, n) > 0 ? true : false;
        }

        initialBlockIdx = initialBlockIdx + weightBlockSize;

    }

    if(fs.is_open()) {
        // Write the header
        fs.write(reinterpret_cast<const char *>(&this->_numOfNodes), this->_headerBlockSize);
        fs.write(reinterpret_cast<const char *>(&this->_dim), this->_headerBlockSize);

        for(unsigned int row=0; row< this->_numOfNodes; row++) {
            vector<uint8_t> bin(this->_dim/8, 0);
            for (unsigned int d = 0; d < this->_dim; d++) {
                bin[int(d/8)] <<= 1;
                if (embMatrix[row][d])
                    bin[int(d/8)] += 1;
            }
            copy(bin.begin(), bin.end(), std::ostreambuf_iterator<char>(fs));
        }

        for(unsigned int node=0; node < this->_numOfNodes; node++)
            delete [] embMatrix[node];
        delete [] embMatrix;

        fs.close();

    } else {
        cout << "An error occurred during opening the file!" << endl;
    }

}






#endif //MODEL_H
