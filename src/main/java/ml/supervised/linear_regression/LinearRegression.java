package ml.supervised.linear_regression;

import static ml.util.Util.matrixToString;
import static ml.util.Util.toOneDimArray;

import java.io.IOException;
import java.util.List;

import ml.supervised.knn.FileHelper;
import Jama.LUDecomposition;
import Jama.Matrix;

public class LinearRegression {

    static final String INPUT_FILE_NAME = "/LinearRegressionDataSet.txt";

    public static void main(String[] args) throws IOException {
        FileHelper fileHelper = new FileHelper(INPUT_FILE_NAME);
        Matrix inputValues = fileHelper.getMatrix();
        List<Double> outputValues = fileHelper.getOutputValues();
        LinearRegression linearRegression = new LinearRegression();
        Matrix ws = linearRegression.standRegres(inputValues, outputValues);
        System.out.println(matrixToString(ws));

        Matrix predicted = inputValues.times(ws);
        System.out.println(matrixToString(predicted));
    }

    public Matrix standRegres(Matrix inputValues, List<Double> outputValues) {
        Matrix xTx = inputValues.transpose().times(inputValues);
        if (new LUDecomposition(xTx).det() == 0.0) {
            throw new IllegalArgumentException("This matrix is singular, cannot do inverse");
        }
        Matrix yMat = new Matrix(toOneDimArray(outputValues), 1);
        yMat = yMat.transpose();
        return xTx.inverse().times(inputValues.transpose().times(yMat));
    }

    // def lwlr(testPoint,xArr,yArr,k=1.0):
    // xMat = mat(xArr); yMat = mat(yArr).T
    // m = shape(xMat)[0]
    // weights = mat(eye((m)))
    // for j in range(m):
    // diffMat = testPoint - xMat[j,:]
    // weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    // xTx = xMat.T * (weights * xMat)
    // if linalg.det(xTx) == 0.0:
    // print "This matrix is singular, cannot do inverse"
    // return
    // ws = xTx.I * (xMat.T * (weights * yMat))
    // return testPoint * ws
    // B
    // Create diagonal
    // matrix
    // C
    // Populate weights
    // with exponentially
    // decaying values
    // def lwlrTest(testArr,xArr,yArr,k=1.0):
    // m = shape(testArr)[0]
    // yHat = zeros(m)
    // for i in range(m):
    // yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    // return yHat
}
