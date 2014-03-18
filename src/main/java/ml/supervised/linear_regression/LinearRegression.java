package ml.supervised.linear_regression;

import static ml.util.Util.arrayToMatrix;
import static ml.util.Util.getRow;
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

        Matrix predicted2 = linearRegression.lwlrTest(inputValues, inputValues, outputValues, 0.01);
        System.out.println(matrixToString(predicted2));
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

    // locally weighted linear regression
    public Matrix lwlr(double[] testPoint, Matrix inputValues, List<Double> outputValues, double k) {
        int m = inputValues.getRowDimension();
        Matrix weights = new Matrix(m, m);
        Matrix testPointMatrix = arrayToMatrix(testPoint, 1);
        for (int r = 0; r < m; r++) {
            Matrix diffMat = testPointMatrix.minus(getRow(inputValues, r));
            Matrix times = diffMat.times(diffMat.transpose());
            double weight = Math.exp(times.get(0, 0) / (-2.0 * k * k));
            weights.set(r, r, weight);
        }
        Matrix xTx = inputValues.transpose().times(weights.times(inputValues));
        if (new LUDecomposition(xTx).det() == 0.0) {
            throw new IllegalArgumentException("This matrix is singular, cannot do inverse");
        }
        Matrix yMat = new Matrix(toOneDimArray(outputValues), 1);
        yMat = yMat.transpose();
        Matrix ws = xTx.inverse().times(inputValues.transpose().times(weights.times(yMat)));
        return testPointMatrix.times(ws);
    }

    public Matrix lwlrTest(Matrix testArr, Matrix xArr, List<Double> yArr, double k) {
        int rowsNumber = testArr.getRowDimension();
        Matrix yHat = new Matrix(rowsNumber, 1);
        for (int r = 0; r < rowsNumber; r++) {
            yHat.set(r, 0, lwlr(testArr.getArray()[r], xArr, yArr, k).get(0, 0));
        }
        return yHat;
    }
}
