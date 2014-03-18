package ml.unsupervised.clustering;

import static ml.unsupervised.clustering.KMeansClustering.INPUT_FILE_NAME;

import java.io.IOException;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.ScatterChart;
import javafx.scene.chart.XYChart;
import javafx.stage.Stage;
import Jama.Matrix;

public class DataSetChart extends Application {

    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage stage) throws IOException {
        stage.setTitle("Clustering chart");
        final NumberAxis xAxis = new NumberAxis(-7, 7, 1);
        final NumberAxis yAxis = new NumberAxis(-7, 7, 1);
        final ScatterChart<Number, Number> sc = new ScatterChart<Number, Number>(xAxis, yAxis);
        xAxis.setLabel("X");
        yAxis.setLabel("Y");
        sc.setTitle("Clustering");

        KMeansClustering clustering = new KMeansClustering();
        Matrix dataSet = clustering.loadDataSet(INPUT_FILE_NAME);

        XYChart.Series<Number, Number> centroidsPoints = new XYChart.Series<Number, Number>();
        centroidsPoints.setName("Centroids");
        for (int r = 0; r < dataSet.getRowDimension(); r++) {
            centroidsPoints.getData().add(new XYChart.Data<Number, Number>(dataSet.get(r, 0), dataSet.get(r, 1)));
        }
        sc.getData().add(centroidsPoints);

        Scene scene = new Scene(sc, 900, 800);
        stage.setScene(scene);
        stage.show();
    }
}
