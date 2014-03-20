package ml.supervised.linear_regression;

import static ml.supervised.linear_regression.LinearRegression.INPUT_FILE_NAME;

import java.io.IOException;
import java.util.List;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.ScatterChart;
import javafx.scene.chart.XYChart;
import javafx.stage.Stage;
import ml.util.FileHelper;
import Jama.Matrix;

public class LinearRegressionChart extends Application {

    public static void main(String[] args) {
        launch(args);
    }

    @SuppressWarnings("unchecked")
    @Override
    public void start(Stage stage) throws IOException {
        stage.setTitle("Linear Regression");
        final NumberAxis xAxis = new NumberAxis(-0.1, 1.1, 0.1);
        final NumberAxis yAxis = new NumberAxis(2.8, 5, 0.1);
        final ScatterChart<Number, Number> sc = new ScatterChart<Number, Number>(xAxis, yAxis);
        xAxis.setLabel("X");
        yAxis.setLabel("Y");
        sc.setTitle("Linear Regression");

        FileHelper fileHelper = new FileHelper(INPUT_FILE_NAME);
        Matrix inputValues = fileHelper.getMatrix();
        List<Double> outputValues = fileHelper.getOutputValues();
        LinearRegression linearRegression = new LinearRegression();
        Matrix ws = linearRegression.standardRegression(inputValues, outputValues);
        Matrix predictedValues = inputValues.times(ws);

        XYChart.Series<Number, Number> trainingSetPoints = new XYChart.Series<Number, Number>();
        trainingSetPoints.setName("training set");
        for (int r = 0; r < inputValues.getRowDimension(); r++) {
            trainingSetPoints.getData().add(
                    new XYChart.Data<Number, Number>(inputValues.get(r, 1), outputValues.get(r)));
        }

        XYChart.Series<Number, Number> predictedSetPoints = new XYChart.Series<Number, Number>();
        predictedSetPoints.setName("predicted set");
        for (int r = 0; r < predictedValues.getRowDimension(); r++) {
            predictedSetPoints.getData().add(
                    new XYChart.Data<Number, Number>(inputValues.get(r, 1), predictedValues.get(r, 0)));
        }

        Matrix lwlrPredictedValues = linearRegression.lwlrTest(inputValues, inputValues, outputValues, 0.003);
        XYChart.Series<Number, Number> lwlrPredictedSetPoints = new XYChart.Series<Number, Number>();
        lwlrPredictedSetPoints.setName("lwlr predicted set");
        for (int r = 0; r < lwlrPredictedValues.getRowDimension(); r++) {
            lwlrPredictedSetPoints.getData().add(
                    new XYChart.Data<Number, Number>(inputValues.get(r, 1), lwlrPredictedValues.get(r, 0)));
        }

        sc.getData().addAll(trainingSetPoints, predictedSetPoints, lwlrPredictedSetPoints);

        Scene scene = new Scene(sc, 900, 800);
        stage.setScene(scene);
        stage.show();
    }
}
