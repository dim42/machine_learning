package ml.supervised.knn;

import static org.junit.Assert.assertEquals;

import java.io.IOException;

import org.junit.Test;

public class KNNClassificationTest {

    @Test
    public void testClassifyTest() throws IOException {
        KNNClassification classification = new KNNClassification();
        int errorCount = classification.classifyTest();
        assertEquals(20, errorCount);
    }
}
