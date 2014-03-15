package ml.unsupervised.clustering;

import java.io.IOException;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.ScatterChart;
import javafx.scene.chart.XYChart;
import javafx.stage.Stage;
import Jama.Matrix;

public class ScatterChartSample2 extends Application {

    @Override
    public void start(Stage stage) throws IOException {
        stage.setTitle("Scatter Chart Sample");
        final NumberAxis xAxis = new NumberAxis(-10, 10, 1);
        final NumberAxis yAxis = new NumberAxis(-10, 10, 1);
        final ScatterChart<Number, Number> sc = new ScatterChart<Number, Number>(xAxis, yAxis);
        xAxis.setLabel("Age (years)");
        yAxis.setLabel("Returns to date");
        sc.setTitle("Clusterring");

        KMeansClustering clustering = new KMeansClustering();
        Matrix dataSet = clustering.loadDataSet("/clusteringTestSet.txt");
        Result kMeans = clustering.kMeans(dataSet, 4);

        Matrix centroids = kMeans.getCentroids();
        XYChart.Series series1 = new XYChart.Series();
        series1.setName("Centroids");
        for (int r = 0; r < centroids.getRowDimension(); r++) {
            series1.getData().add(new XYChart.Data(centroids.get(r, 0), centroids.get(r, 1)));
        }

        Matrix clusterAssment = kMeans.getClusterAssment();
        XYChart.Series series2 = new XYChart.Series();
        series2.setName("Cluster assment");
        for (int r = 0; r < clusterAssment.getRowDimension(); r++) {
            series2.getData().add(new XYChart.Data(clusterAssment.get(r, 0), clusterAssment.get(r, 1)));
        }

        sc.getData().addAll(series1, series2);
        Scene scene = new Scene(sc, 500, 400);
        stage.setScene(scene);
        stage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}
