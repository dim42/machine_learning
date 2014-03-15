package ml.unsupervised.clustering;

import Jama.Matrix;

public class Result {
    private final Matrix centroids;
    private final Matrix clusterAssment;

    public Result(Matrix centroids, Matrix clusterAssment) {
        this.centroids = centroids;
        this.clusterAssment = clusterAssment;
    }

    public Matrix getCentroids() {
        return centroids;
    }

    public Matrix getClusterAssment() {
        return clusterAssment;
    }
}
