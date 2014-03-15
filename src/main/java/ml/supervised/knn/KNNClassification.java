package ml.supervised.knn;

import static ml.supervised.knn.Util.argsort;
import static ml.supervised.knn.Util.arrayToMatrix;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import Jama.Matrix;

public class KNNClassification {
    private static final String INPUT_FILE_NAME = "/KNNClassificationDataSet.txt";
    private static final double TEST_RATIO = 0.5;
    private static final int K_NUMBER = 3;

    public static void main(String[] args) throws IOException {
        KNNClassification classification = new KNNClassification();
        classification.classifyTest();
    }

    /**
     * One common task in machine learning is evaluating an algorithm’s accuracy. One way you can use the existing data
     * is to take some portion, say 90%, to train the classifier. Then you’ll take the remaining 10% to test the
     * classifier and see how accurate it is.
     * 
     * @throws IOException
     */
    public void classifyTest() throws IOException {
        FileHelper fileHelper = new FileHelper(INPUT_FILE_NAME);
        Matrix dataSet = fileHelper.getMatrix();
        Matrix normMat = autoNormalize(dataSet);
        int testSetLength = (int) (normMat.getRowDimension() * TEST_RATIO);
        Matrix trainingSubmatrix = normMat.getMatrix(testSetLength, normMat.getRowDimension() - 1, 0,
                normMat.getColumnDimension() - 1);
        List<Integer> classLabels = fileHelper.getClassLabels();
        List<Integer> trainingClassLabels = classLabels.subList(testSetLength, classLabels.size());

        int errorCount = 0;
        for (int i = 0; i < testSetLength; i++) {
            int classifierResult = classify(normMat.getArray()[i], trainingSubmatrix, trainingClassLabels, K_NUMBER);
            System.out.println(String.format("The classifier came back with: %d, the real answer is: %d",
                    classifierResult, classLabels.get(i)));
            if (classifierResult != classLabels.get(i)) {
                errorCount++;
            }
        }

        System.out.println("The total error rate is: " + errorCount / (float) testSetLength);
        System.out.println("Error number: " + errorCount);
        // The classifier came back with: 1, the real answer is: 1
        // The classifier came back with: 3, the real answer is: 3
        // The classifier came back with: 1, the real answer is: 1
        // The classifier came back with: 2, the real answer is: 1
        // The classifier came back with: 2, the real answer is: 2
        // The classifier came back with: 1, the real answer is: 1
        // The classifier came back with: 1, the real answer is: 1
        // The classifier came back with: 2, the real answer is: 2
        // The total error rate is: 0.068
        // Error number: 34
    }

    public int classify(double[] inX, Matrix dataSet, List<Integer> labels, int k) {
        Matrix inXmatrix = arrayToMatrix(inX, dataSet.getRowDimension());
        Matrix diffMat = inXmatrix.minus(dataSet);
        Matrix sqDiffMat = diffMat.arrayTimes(diffMat);
        List<Double> distances = getSqDistances(sqDiffMat);
        Integer[] sortedDistancesIndices = argsort(distances);
        Map<Integer, Integer> classCount = getClassCountMap(labels, k, sortedDistancesIndices);
        return getMaxVotedClass(classCount);
    }

    private List<Double> getSqDistances(Matrix sqDiffMat) {
        List<Double> distances = new ArrayList<>();
        for (int i = 0; i < sqDiffMat.getRowDimension(); i++) {
            double[] row = sqDiffMat.getArray()[i];
            double rowSum = 0;
            for (int j = 0; j < row.length; j++) {
                rowSum += row[j];
            }
            distances.add(Math.sqrt(rowSum));
        }
        return distances;
    }

    private int getMaxVotedClass(Map<Integer, Integer> classCount) {
        int maxVotedClass = -1;
        for (Entry<Integer, Integer> entry : classCount.entrySet()) {
            if (entry.getValue() > maxVotedClass) {
                maxVotedClass = entry.getKey();
            }
        }
        return maxVotedClass;
    }

    private Map<Integer, Integer> getClassCountMap(List<Integer> labels, int k, Integer[] sortedDistancesIndices) {
        Map<Integer, Integer> result = new LinkedHashMap<>();
        for (int i = 0; i < k; i++) {
            int classToVote = labels.get(sortedDistancesIndices[i]);
            Integer count = result.get(classToVote);
            result.put(classToVote, count == null ? 1 : count + 1);
        }
        return result;
    }

    public Matrix autoNormalize(Matrix dataSet) {
        double[] minVals = new double[dataSet.getColumnDimension()];
        Arrays.fill(minVals, Double.MAX_VALUE);
        double[] maxVals = new double[dataSet.getColumnDimension()];
        Arrays.fill(maxVals, Double.MIN_VALUE);
        for (int i = 0; i < dataSet.getRowDimension(); i++) {
            for (int j = 0; j < dataSet.getColumnDimension(); j++) {
                double el0 = dataSet.get(i, j);
                if (el0 < minVals[j]) {
                    minVals[j] = el0;
                }
                if (el0 > maxVals[j]) {
                    maxVals[j] = el0;
                }
            }
        }
        Matrix minMx = arrayToMatrix(minVals, dataSet.getRowDimension());
        Matrix maxMx = arrayToMatrix(maxVals, dataSet.getRowDimension());
        Matrix range = maxMx.minus(minMx);
        Matrix diff = dataSet.minus(minMx);
        return diff.arrayRightDivide(range);
    }
}
