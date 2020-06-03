#include "gtest/gtest.h"
#include "../lib/Matrix.h"
#include "../src/Matrix.cpp"
#include <vector>

using namespace std;

TEST(Initialization, Equals) {

    // Generate a vector storing values
    vector<Triplet<double>> baseList;
    baseList.push_back(Triplet<double>{0, 0, 19.0});
    baseList.push_back(Triplet<double>{0,1,8.7});
    baseList.push_back(Triplet<double>{1,2, 2.3});
    baseList.push_back(Triplet<double>{1,4, 6.2});
    baseList.push_back(Triplet<double>{2, 3, 4.7});
    baseList.push_back(Triplet<double>{3,0, 2.9});
    baseList.push_back(Triplet<double>{4, 3, 12.0});

    // Generate matrix from the vector
    sparseMatrix<double> matrix(5, 5,25);
    for(int n=0; n<baseList.size(); n++)
        matrix.insert(baseList[n].row, baseList[n].column, baseList[n].value);

    vector<Triplet<double>> &refList = matrix.getMat();
    for(int n=0; n<baseList.size(); n++) {
        EXPECT_EQ(baseList[n].row, refList[n].row);
        EXPECT_EQ(baseList[n].column, refList[n].column);
        EXPECT_EQ(baseList[n].value, refList[n].value);
    }
}

TEST(Assign, Equals) {

    sparseMatrix<double> baseMat(5, 5,25);
    baseMat.insert(0, 0, 19.0);
    baseMat.insert(0,1,8.7);
    baseMat.insert(1,2, 2.3);
    baseMat.insert(1,4, 6.2);
    baseMat.insert(2, 3, 4.7);
    baseMat.insert(3,0, 2.9);
    baseMat.insert(4, 3, 12.0);

    sparseMatrix<double> refMat(5, 5,25);

    refMat.assign(baseMat);

    vector<Triplet<double>> &baseList = baseMat.getMat();
    vector<Triplet<double>> &refList = refMat.getMat();
    for(int n=0; n<baseList.size(); n++) {
        EXPECT_EQ(baseList[n].row, refList[n].row);
        EXPECT_EQ(baseList[n].column, refList[n].column);
        EXPECT_EQ(baseList[n].value, refList[n].value);
    }
}

TEST(Addition, Equals) {

    EXPECT_EQ(1, 1);

}

TEST(Transpose, Equals) {

    sparseMatrix<double> baseMat(5, 5,25);
    baseMat.insert(0, 0, 19.0);
    baseMat.insert(0,1,8.7);
    baseMat.insert(1,2, 2.3);
    baseMat.insert(1,4, 6.2);
    baseMat.insert(2, 3, 4.7);
    baseMat.insert(3,0, 2.9);
    baseMat.insert(4, 3, 12.0);

    sparseMatrix<double> refMat(5, 5,25);
    refMat.insert(0, 0, 19.0);
    refMat.insert(0,3, 2.9);
    refMat.insert(1,0,8.7);
    refMat.insert(2,1, 2.3);
    refMat.insert(3, 2, 4.7);
    refMat.insert(3, 4, 12.0);
    refMat.insert(4,1, 6.2);

    vector<Triplet<double>> &baseList = baseMat.getMat();

    // Take the transpose of the matrix
    refMat.transpose();
    vector<Triplet<double>> &refList = refMat.getMat();
    /*
    for(int n=0; n<baseList.size(); n++) {
        EXPECT_EQ(baseList[n].value, refList[n].value);
    }
    */

}