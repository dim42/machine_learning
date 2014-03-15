package ml.unsupervised.clustering;

import static ml.supervised.knn.Util.getReader;
import static ml.supervised.knn.Util.listToArray;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import Jama.Matrix;

public class KMeansClustering {

    public static void main(String[] args) throws IOException {
        KMeansClustering clustering = new KMeansClustering();
        Matrix dataSet = clustering.loadDataSet("/clusteringTestSet.txt");
        clustering.kMeans(dataSet, 4);
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

    public Matrix kMeans(Matrix dataSet, int k) {
        Matrix clusterAssment = new Matrix(dataSet.getRowDimension(), 2);
        Matrix centroids = randCent(dataSet, k);
        boolean clusterChanged = true;
        while (clusterChanged) {
            clusterChanged = false;
            for (int i = 0; i < dataSet.getRowDimension(); i++) {
                double minDist = Double.MAX_VALUE;
                int minIndex = -1;
                for (int j = 0; j < k; j++) {
                    double distJI = distEclud(centroids.getArray()[j], dataSet.getArray()[i]);
                    if (distJI < minDist) {
                        minDist = distJI;
                        minIndex = j;
                    }
                }
                if (clusterAssment.get(i, 0) != minIndex) {
                    clusterChanged = true;
                }
                clusterAssment.set(i, 0, minIndex);
                clusterAssment.set(i, 1, minDist * minDist);
            }
            System.out.println("centroids: " + Arrays.deepToString(centroids.getArray()));
            for (int i = 0; i < k; i++) {
                // ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
                // centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean
            }
        }
        System.out.println("clusterAssment: " + Arrays.deepToString(clusterAssment.getArray()));
        System.out.println("centroids: " + Arrays.deepToString(centroids.getArray()));
        return centroids;
        // return centroids, clusterAssment
    }

    private Matrix loadDataSet(String fileName) throws IOException {
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

    public Matrix randCent(Matrix dataSet, int k) {
        int n = dataSet.getColumnDimension();
        Matrix centroids = new Matrix(k, n);
        for (int i = 0; i < n; i++) {
            Matrix column = getColumn(dataSet, i);
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
            for (int j = 0; j < columnArray.length; j++) {
                columnArray[j] = min + range * Math.random();
            }
        }
        return centroids;
    }

    private Matrix getColumn(Matrix dataSet, int i) {
        return dataSet.getMatrix(0, dataSet.getRowDimension() - 1, i, i);
    }
}
