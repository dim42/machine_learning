package ml.supervised.knn;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
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
class FileHelper {

    private final Matrix matrix;
    private final List<Integer> classLabels;

    public FileHelper(String fileName) throws IOException {
        matrix = file2matrix(fileName);
        classLabels = getClassLabels(fileName);
    }

    public Matrix getMatrix() {
        return matrix;
    }

    public List<Integer> getClassLabels() {
        return classLabels;
    }

    private Matrix file2matrix(String fileName) throws IOException {
        int numberOfLines = getNumberOfLines(fileName);
        double[][] result = null;
        try (BufferedReader reader = getReader(fileName)) {
            String line;
            String[] values;
            int lineNumber = 0;
            boolean first = true;
            while ((line = reader.readLine()) != null) {
                values = line.split("\t");
                if (first) {
                    int numberOfFeatures = values.length - 1;
                    result = new double[numberOfLines][numberOfFeatures];
                    first = false;
                }
                for (int i = 0; i < values.length - 1; i++) {
                    result[lineNumber][i] = Double.parseDouble(values[i]);
                }
                lineNumber++;
            }
        }
        return new Matrix(result);
    }

    private List<Integer> getClassLabels(String fileName) throws FileNotFoundException, IOException {
        List<Integer> classLabels = new ArrayList<>();
        try (BufferedReader reader = getReader(fileName)) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] split = line.split("\t");
                classLabels.add(Integer.parseInt(split[split.length - 1]));
            }
        }
        return classLabels;
    }

    private int getNumberOfLines(String fileName) throws IOException, FileNotFoundException {
        try (BufferedReader reader = getReader(fileName)) {
            int numberOfLines = 0;
            while (reader.readLine() != null) {
                numberOfLines++;
            }
            return numberOfLines;
        }
    }

    private BufferedReader getReader(String fileName) throws FileNotFoundException {
        return new BufferedReader(new InputStreamReader(KNNClassification.class.getResourceAsStream(fileName)));
    }
}