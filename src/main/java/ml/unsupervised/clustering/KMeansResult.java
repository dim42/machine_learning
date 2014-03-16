package ml.unsupervised.clustering;

import static ml.util.Util.matrixToString;
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
        return "KMeansResult [centroids=" + matrixToString(centroids) + ",\nclusterAssment="
                + matrixToString(clusterAssment) + "]";
    }
}
