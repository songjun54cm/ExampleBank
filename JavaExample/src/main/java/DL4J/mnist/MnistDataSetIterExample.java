package DL4J.mnist;

import org.deeplearning4j.datasets.iterator.BaseDatasetIterator;
import org.nd4j.linalg.dataset.api.iterator.fetcher.BaseDataFetcher;

import java.io.IOException;

/**
 * Author: songjun
 * Date: 2019/1/4
 * Description:
 **/
public class MnistDataSetIterExample extends BaseDatasetIterator {
    public MnistDataSetIterExample(int batch, int numExamples, BaseDataFetcher fetcher) {
        super(batch, numExamples, fetcher);
    }

    public MnistDataSetIterExample(int batch, String feaDataPath, String labelDataPath) throws IOException {
        BaseDataFetcher fetcher = new MnistDataFetcherExample(feaDataPath, labelDataPath);
        int numExample = fetcher.getNumExample();
        super(batch, numExample, fetcher);
    }
}


