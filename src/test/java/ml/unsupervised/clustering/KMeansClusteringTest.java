package ml.unsupervised.clustering;

import static ml.unsupervised.clustering.KMeansClustering.CLUSTER_NUMBER;
import static ml.unsupervised.clustering.KMeansClustering.INPUT_FILE_NAME;

import java.io.IOException;

import org.junit.Test;

import Jama.Matrix;

public class KMeansClusteringTest {

    @Test
    public void testKMeans() throws IOException {
        KMeansClustering clustering = new KMeansClustering();
        Matrix dataSet = clustering.loadDataSet(INPUT_FILE_NAME);

        KMeansResult result = clustering.kMeans(dataSet, CLUSTER_NUMBER);
        System.out.println(result);

        result = clustering.biKmeans(dataSet, CLUSTER_NUMBER);
        System.out.println(result);
    }
}
