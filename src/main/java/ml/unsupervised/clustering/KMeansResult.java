package ml.unsupervised.clustering;

import java.util.Arrays;

import Jama.Matrix;

public class KMeansResult {
    private final Matrix centroids;
    private final Matrix clusterAssment;

    public KMeansResult(Matrix centroids, Matrix clusterAssment) {
        this.centroids = centroids;
        this.clusterAssment = clusterAssment;
    }

    public Matrix getCentroids() {
        return centroids;
    }

    public Matrix getClusterAssment() {
        return clusterAssment;
    }

    @Override
    public String toString() {
        return "KMeansResult [centroids=" + Arrays.deepToString(centroids.getArray()) + ",\nclusterAssment="
                + Arrays.deepToString(clusterAssment.getArray()) + "]";
    }
}
