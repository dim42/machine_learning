package ml.supervised.linear_regression;

import static ml.supervised.knn.Util.getReader;
import static ml.supervised.knn.Util.listToArray;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import Jama.Matrix;

public class LinearRegression {
    public static void main(String[] args) {

    }

    private Matrix loadDataSet(String fileName) throws IOException {
        List<double[]> list = new ArrayList<>();
        try (BufferedReader reader = getReader(fileName)) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] values = line.split("\t");
                double[] row = new double[values.length];
                for (int i = 0; i < values.length; i++) {
                    row[i] = Double.parseDouble(values[i]);
                }
                list.add(row);
            }
        }
        return new Matrix(listToArray(list));
    }

    // def loadDataSet(fileName):
    // numFeat = len(open(fileName).readline().split('\t')) - 1
    // dataMat = []; labelMat = []
    // fr = open(fileName)
    // for line in fr.readlines():
    // lineArr =[]
    // curLine = line.strip().split('\t')
    // for i in range(numFeat):
    // Finding best-fit lines with linear regression
    // 157
    // lineArr.append(float(curLine[i]))
    // dataMat.append(lineArr)
    // labelMat.ppend(float(curLine[-1]))
    // return dataMat,labelMat
    //
    // public void standRegres(xArr,yArr){
    // xMat = mat(xArr); yMat = mat(yArr).T
    // xTx = xMat.T*xMat
    // if linalg.det(xTx) == 0.0:
    // print "This matrix is singular, cannot do inverse"
    // return
    // }

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
