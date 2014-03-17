package ml.supervised.linear_regression;

import static ml.util.Util.matrixToString;
import static ml.util.Util.toOneDimArray;

import java.io.IOException;
import java.util.ArrayList;
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

    public Matrix lwlr(List<Double> testPoint, Matrix inputValues, List<Double> outputValues, int k) {
        int m = inputValues.getRowDimension();
        Matrix weights = null;
        // weights = mat(eye((m)))
        for (int i = 0; i < m; i++) {
            // diffMat = testPoint - inputValues[j,:]
            // weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
        }
        Matrix xTx = inputValues.transpose().times(weights.times(inputValues));
        if (new LUDecomposition(xTx).det() == 0.0) {
            throw new IllegalArgumentException("This matrix is singular, cannot do inverse");
        }
        Matrix yMat = new Matrix(toOneDimArray(outputValues), 1);
        yMat = yMat.transpose();
        Matrix ws = xTx.inverse().times(inputValues.transpose().times(weights.times(yMat)));
        // return testPoint * ws
        return null;
    }

    public List<Double> lwlrTest(Matrix testArr, Matrix xArr, List<Double> yArr, int k) {
        int m = testArr.getRowDimension();
        List<Double> yHat = new ArrayList<>(m);
        for (int j = 0; j < m; j++) {
            // yHat[i] = lwlr(testArr[i],xArr,yArr,k)
        }
        return yHat;
    }
}
