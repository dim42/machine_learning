package ml.supervised.knn;

import static ml.util.Util.getReader;
import static ml.util.Util.toTwoDimArray;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import Jama.Matrix;

// 40920 8.326976 0.953952 3
// 14488 7.153469 1.673904 2
// 26052 1.441871 0.805124 1
// 75136 13.147394 0.428964 1
// 38344 1.669788 0.134296 1
// 72993 10.141740 1.032955 1
// 35948 6.830792 1.213192 3
// 42666 13.276369 0.543880 3
public class FileHelper {

    private final Matrix matrix;
    private final List<Double> classLabels = new ArrayList<>();

    public FileHelper(String fileName) throws IOException {
        matrix = loadDataSet(fileName);
    }

    public Matrix getMatrix() {
        return matrix;
    }

    public List<Integer> getClassLabels() {
        List<Integer> result = new ArrayList<>();
        for (Double val : classLabels) {
            result.add(val.intValue());
        }
        return result;
    }

    public List<Double> getOutputValues() {
        return classLabels;
    }

    private Matrix loadDataSet(String fileName) throws IOException {
        List<double[]> list = new ArrayList<>();
        try (BufferedReader reader = getReader(fileName)) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] values = line.split("\t");
                double[] row = new double[values.length - 1];
                for (int i = 0; i < values.length - 1; i++) {
                    row[i] = Double.parseDouble(values[i]);
                }
                list.add(row);
                classLabels.add(Double.parseDouble(values[values.length - 1]));
            }
        }
        return new Matrix(toTwoDimArray(list));
    }
}