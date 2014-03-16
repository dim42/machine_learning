package ml.unsupervised.clustering;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

public class PointsByClusters {
    Map<Integer, List<double[]>> map = new LinkedHashMap<>();

    public Map<Integer, List<double[]>> getMap() {
        return map;
    }

    public void put(int clN, List<double[]> clusterPoints) {
        map.put(clN, clusterPoints);
    }

    @Override
    public String toString() {
        StringBuilder result = new StringBuilder("PointsByClusters:\n");
        for (Entry<Integer, List<double[]>> entry : map.entrySet()) {
            result.append("Cluster: " + entry.getKey()).append(", points: ");
            for (double[] point : entry.getValue()) {
                result.append("x=" + point[0] + ", y=" + point[1] + "; ");
            }
            result.append("\n");
        }
        return result.toString();
    }
}
