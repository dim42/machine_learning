package ml.unsupervised.clustering;

import static ml.util.Util.getColumn;
import static ml.util.Util.getReader;
import static ml.util.Util.mean;
import static ml.util.Util.toTwoDimArray;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import Jama.Matrix;

public class KMeansClustering {

    static final String INPUT_FILE_NAME = "/clusteringTestSet.txt";
    static final int CLUSTER_NUMBER = 4;

    public static void main(String[] args) throws IOException {
        KMeansClustering clustering = new KMeansClustering();
        Matrix dataSet = clustering.loadDataSet(INPUT_FILE_NAME);

        KMeansResult result = clustering.kMeans(dataSet, CLUSTER_NUMBER);
        System.out.println("result: " + result);

        PointsByClusters pointsByClusters = clustering.pointsByClusters(dataSet, result.getClusterAssment());
        System.out.println(pointsByClusters);
    }

    public KMeansResult biKmeans(Matrix dataSet, int clusterNumber) {
        // Initially create one cluster
        Matrix clusterAssment = new Matrix(dataSet.getRowDimension(), 2);
        double[] xColumnArray = getColumn(dataSet, 0).getColumnPackedCopy();
        double xMean = mean(xColumnArray);
        double[] yColumnArray = getColumn(dataSet, 1).getColumnPackedCopy();
        double yMean = mean(yColumnArray);
        double[] centroid0 = new double[] { xMean, yMean };
        List<double[]> centList = new ArrayList<>();
        centList.add(centroid0);
        for (int r = 0; r < dataSet.getRowDimension(); r++) {
            double dist = distEclud(centroid0, dataSet.getArray()[r]);
            clusterAssment.set(r, 1, dist * dist);
        }

        while (centList.size() < clusterNumber) {
            double lowestSSE = Double.MAX_VALUE;
            int bestCentToSplit = -1;
            Matrix bestNewCents = null;
            Matrix bestClustAss = null;
            for (int clusterIndex = 0; clusterIndex < centList.size(); clusterIndex++) {
                List<double[]> currPoints = new ArrayList<>();
                for (int r = 0; r < clusterAssment.getRowDimension(); r++) {
                    if (clusterAssment.get(r, 0) == clusterIndex) {
                        currPoints.add(new double[] { dataSet.get(r, 0), dataSet.get(r, 1) });
                    }
                }
                if (currPoints.isEmpty())
                    continue;
                Matrix ptsInCurrCluster = new Matrix(toTwoDimArray(currPoints));

                KMeansResult kMeansResult = kMeans(ptsInCurrCluster, 2);
                double sseSplit = 0;
                for (int r = 0; r < kMeansResult.getClusterAssment().getRowDimension(); r++) {
                    sseSplit += kMeansResult.getClusterAssment().get(r, 1);
                }

                double sseNotSplit = 0;
                for (int r = 0; r < clusterAssment.getRowDimension(); r++) {
                    if (clusterAssment.get(r, 0) != clusterIndex) {
                        sseNotSplit += clusterAssment.get(r, 1);
                    }
                }
                System.out.println("sseSplit: " + sseSplit);
                System.out.println("sseNotSplit: " + sseNotSplit);

                if (sseSplit + sseNotSplit < lowestSSE) {
                    bestCentToSplit = clusterIndex;
                    bestNewCents = kMeansResult.getCentroids();
                    bestClustAss = kMeansResult.getClusterAssment().copy();
                    lowestSSE = sseSplit + sseNotSplit;
                }
            }
            // Update the cluster assignments
            for (int r = 0; r < bestClustAss.getRowDimension(); r++) {
                double val = bestClustAss.get(r, 0);
                if (val == 1) {
                    bestClustAss.set(r, 0, centList.size());
                } else if (val == 0) {
                    bestClustAss.set(r, 0, bestCentToSplit);
                }
            }
            System.out.println("the bestCentToSplit is: " + bestCentToSplit);
            System.out.println("the length of bestClustAss is: " + bestClustAss.getRowDimension());

            centList.set(bestCentToSplit, bestNewCents.getArray()[0]);
            centList.add(bestNewCents.getArray()[1]);
            // bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
            // for (int r = 0; r < clusterAssment.getRowDimension(); r++) {
            // if ((int) clusterAssment.get(r, 0) == bestCentToSplit) {
            // clusterAssment.set(r, 0, bestClustAss.get(r, 0));
            // }
            // }

            int ind = 0;
            for (int r = 0; r < clusterAssment.getRowDimension(); r++) {
                if ((int) clusterAssment.get(r, 0) == bestCentToSplit) {
                    clusterAssment.set(r, 0, bestClustAss.get(ind++, 0));
                }
            }
            // clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss
        }
        return new KMeansResult(new Matrix(toTwoDimArray(centList)), clusterAssment);
    }

    public KMeansResult kMeans(Matrix dataSet, int clusterNumber) {
        Matrix clusterAssment = new Matrix(dataSet.getRowDimension(), 2);
        Matrix centroids = generateRandomCentroids(dataSet, clusterNumber);
        boolean clusterChanged = true;
        while (clusterChanged) {
            clusterChanged = false;
            for (int r = 0; r < dataSet.getRowDimension(); r++) {
                double minDist = Double.MAX_VALUE;
                int minIndex = -1;
                for (int j = 0; j < clusterNumber; j++) {
                    double distJI = distEclud(centroids.getArray()[j], dataSet.getArray()[r]);
                    if (distJI < minDist) {
                        minDist = distJI;
                        minIndex = j;
                    }
                }
                if (clusterAssment.get(r, 0) != minIndex) {
                    clusterChanged = true;
                }
                clusterAssment.set(r, 0, minIndex);
                clusterAssment.set(r, 1, minDist * minDist);
            }

            // Update centroid location
            for (int clN = 0; clN < clusterNumber; clN++) {
                List<Double> clusterPointsX = new ArrayList<>();
                List<Double> clusterPointsY = new ArrayList<>();
                for (int r = 0; r < clusterAssment.getRowDimension(); r++) {
                    if (clusterAssment.get(r, 0) == clN) {
                        clusterPointsX.add(dataSet.get(r, 0));
                        clusterPointsY.add(dataSet.get(r, 1));
                    }
                }
                centroids.set(clN, 0, mean(clusterPointsX));
                centroids.set(clN, 1, mean(clusterPointsY));
            }
        }
        return new KMeansResult(centroids, clusterAssment);
    }

    public PointsByClusters pointsByClusters(Matrix dataSet, Matrix clusterAssment) {
        PointsByClusters result = new PointsByClusters();
        for (int clusterIndex = 0; clusterIndex < clusterNumber(clusterAssment); clusterIndex++) {
            List<double[]> clusterPoints = new ArrayList<>();
            for (int r = 0; r < clusterAssment.getRowDimension(); r++) {
                if (clusterAssment.get(r, 0) == clusterIndex) {
                    double[] point = dataSet.getArray()[r];
                    clusterPoints.add(point);
                }
            }
            result.put(clusterIndex, clusterPoints);
        }
        return result;
    }

    private int clusterNumber(Matrix clusterAssment) {
        Set<Integer> set = new HashSet<>();
        for (int r = 0; r < clusterAssment.getRowDimension(); r++) {
            set.add((int) clusterAssment.get(r, 0));
        }
        return set.size();
    }

    Matrix loadDataSet(String fileName) throws IOException {
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
        return new Matrix(toTwoDimArray(list));
    }

    public double distEclud(double[] vecA, double[] vecB) {
        double sum = 0;
        for (int i = 0; i < vecA.length; i++) {
            sum += Math.pow(vecA[i] - vecB[i], 2);
        }
        return Math.sqrt(sum);
    }

    public Matrix generateRandomCentroids(Matrix dataSet, int k) {
        int n = dataSet.getColumnDimension();
        Matrix centroids = new Matrix(k, n);
        for (int c = 0; c < n; c++) {
            Matrix column = getColumn(dataSet, c);
            double[] columnArray = column.getColumnPackedCopy();
            double min = Double.MAX_VALUE;
            double max = Double.MIN_VALUE;
            for (int j = 0; j < columnArray.length; j++) {
                double el = columnArray[j];
                if (el < min) {
                    min = el;
                }
                if (el > max) {
                    max = el;
                }
            }
            double range = max - min;
            for (int r = 0; r < centroids.getRowDimension(); r++) {
                centroids.set(r, c, min + range * Math.random());
            }
        }
        return centroids;
    }
}
