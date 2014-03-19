package ml.util;

import static ml.util.Util.getReader;
import static ml.util.Util.toTwoDimArray;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import Jama.Matrix;

public class FileHelper {

    private final Matrix matrix;
    private final List<Double> classLabels = new ArrayList<>();

    public FileHelper(String fileName) throws IOException {
        matrix = loadDataSet(fileName);
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
}