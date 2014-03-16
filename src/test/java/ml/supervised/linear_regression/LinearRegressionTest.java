package ml.supervised.linear_regression;

import static ml.supervised.linear_regression.LinearRegression.INPUT_FILE_NAME;
import static org.junit.Assert.assertArrayEquals;

import java.io.IOException;
import java.util.List;

import ml.supervised.knn.FileHelper;

import org.junit.Test;

import Jama.Matrix;

public class LinearRegressionTest {

    @Test
    public void testStandRegres() throws IOException {
        FileHelper fileHelper = new FileHelper(INPUT_FILE_NAME);
        Matrix inputValues = fileHelper.getMatrix();
        List<Double> outputValues = fileHelper.getOutputValues();
        LinearRegression linearRegression = new LinearRegression();

        Matrix ws = linearRegression.standRegres(inputValues, outputValues);

        assertArrayEquals(new Double[][] { { 3.007743242697588 }, { 1.6953226421712237 } }, ws.getArray());
    }

}
