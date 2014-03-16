package ml.unsupervised.clustering;

import static ml.supervised.knn.Util.getReader;
import static ml.supervised.knn.Util.listToArray;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import Jama.Matrix;

public class KMeansClustering {

    private static final int CLUSTER_NUMBER = 4;

    public static void main(String[] args) throws IOException {
        KMeansClustering clustering = new KMeansClustering();
        Matrix dataSet = clustering.loadDataSet("/clusteringTestSet.txt");

        KMeansResult result = clustering.kMeans(dataSet, CLUSTER_NUMBER);
        System.out.println("result: " + result);

        PointsByClusters pointsByClusters = clustering.pointsByClusters(dataSet, result.getClusterAssment(),
                CLUSTER_NUMBER);
        System.out.println(pointsByClusters);
    }

    public Matrix biKmeans(Matrix dataSet, int k) {
        Matrix clusterAssment = new Matrix(dataSet.getRowDimension(), 2);
        // centroid0 = mean(dataSet, axis=0).tolist()[0]
        // centList =[centroid0] #create a list with one centroid
        for (int i = 0; i < dataSet.getRowDimension(); i++) {
            // clusterAssment[j,1] = distEclud(mat(centroid0), dataSet[j,:])**2
        }
        // while (len(centList) < k):
        // lowestSSE = inf
        // for i in range(len(centList)):
        // ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]#get the data points currently in cluster i
        // centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2)
        // sseSplit = sum(splitClustAss[:,1])#compare the SSE to the currrent minimum
        // sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
        // print "sseSplit, and notSplit: ",sseSplit,sseNotSplit
        // if (sseSplit + sseNotSplit) < lowestSSE:
        // bestCentToSplit = i
        // bestNewCents = centroidMat
        // bestClustAss = splitClustAss.copy()
        // lowestSSE = sseSplit + sseNotSplit
        // bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        // bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        // print 'the bestCentToSplit is: ',bestCentToSplit
        // print 'the len of bestClustAss is: ', len(bestClustAss)
        // centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids
        // centList.append(bestNewCents[1,:].tolist()[0])
        // clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters,
        // and SSE
        // return mat(centList), clusterAssment
        return clusterAssment;
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

    public PointsByClusters pointsByClusters(Matrix dataSet, Matrix clusterAssment, int clusterNumber) {
        PointsByClusters result = new PointsByClusters();
        for (int clN = 0; clN < clusterNumber; clN++) {
            List<double[]> clusterPoints = new ArrayList<>();
            for (int r = 0; r < clusterAssment.getRowDimension(); r++) {
                if (clusterAssment.get(r, 0) == clN) {
                    double[] point = dataSet.getArray()[r];
                    clusterPoints.add(point);
                }
            }
            result.put(clN, clusterPoints);
        }
        return result;
    }

    private double mean(List<Double> list) {
        double sum = 0;
        for (Double val : list) {
            sum += val;
        }
        return sum / list.size();
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
        return new Matrix(listToArray(list));
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

    private Matrix getColumn(Matrix dataSet, int i) {
        return dataSet.getMatrix(0, dataSet.getRowDimension() - 1, i, i);
    }
}
