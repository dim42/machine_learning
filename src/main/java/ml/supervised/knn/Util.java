package ml.supervised.knn;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import Jama.Matrix;

public class Util {

    public static Matrix arrayToMatrix(double[] input, int rowDimension) {
        double[][] result = new double[rowDimension][input.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = input;
        }
        return new Matrix(result);
    }

    public static double[][] listToArray(List<double[]> list) {
        double[][] array = new double[list.size()][];
        for (int i = 0; i < list.size(); i++) {
            array[i] = list.get(i);
        }
        return array;
    }

    public static BufferedReader getReader(String fileName) throws FileNotFoundException {
        return new BufferedReader(new InputStreamReader(KNNClassification.class.getResourceAsStream(fileName)));
    }

    public static Integer[] argsort(final List<Double> a) {
        Integer[] indexes = new Integer[a.size()];
        for (int i = 0; i < indexes.length; i++) {
            indexes[i] = i;
        }
        Arrays.sort(indexes, new Comparator<Integer>() {
            @Override
            public int compare(final Integer i1, final Integer i2) {
                return Double.compare(a.get(i1), a.get(i2));
            }
        });
        return indexes;
    }
}
