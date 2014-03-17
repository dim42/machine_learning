package ml.unsupervised.clustering;

import static ml.unsupervised.clustering.KMeansClustering.CLUSTER_NUMBER;
import static ml.unsupervised.clustering.KMeansClustering.INPUT_FILE_NAME;

import java.io.IOException;
import java.util.List;
import java.util.Map.Entry;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.ScatterChart;
import javafx.scene.chart.XYChart;
import javafx.stage.Stage;
import Jama.Matrix;

public class BiClusteringChart extends Application {

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

        KMeansResult kMeans = clustering.biKmeans(dataSet, CLUSTER_NUMBER);
        PointsByClusters pointsByClusters = clustering.pointsByClusters(dataSet, kMeans.getClusterAssment(),
                CLUSTER_NUMBER);

        Matrix centroids = kMeans.getCentroids();
        XYChart.Series<Number, Number> centroidsPoints = new XYChart.Series<Number, Number>();
        centroidsPoints.setName("Centroids");
        for (int r = 0; r < centroids.getRowDimension(); r++) {
            centroidsPoints.getData().add(new XYChart.Data<Number, Number>(centroids.get(r, 0), centroids.get(r, 1)));
        }
        sc.getData().add(centroidsPoints);

        for (Entry<Integer, List<double[]>> entry : pointsByClusters.getMap().entrySet()) {
            XYChart.Series<Number, Number> dataSetPoints = new XYChart.Series<Number, Number>();
            dataSetPoints.setName("Cluster" + entry.getKey());
            if (entry.getValue().size() > 0) {
                for (double[] point : entry.getValue()) {
                    dataSetPoints.getData().add(new XYChart.Data<Number, Number>(point[0], point[1]));
                }
                sc.getData().add(dataSetPoints);
            }
        }

        Scene scene = new Scene(sc, 900, 800);
        stage.setScene(scene);
        stage.show();
    }
}
