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
    public MnistDataSetIterExample(int batch, BaseDataFetcher fetcher) {
        super(batch, -1,  fetcher);
    }

    public int getFeaSize(){
        return fetcher.inputColumns();
    }

    public int getNumSample(){
        return fetcher.totalExamples();
    }

    public int getNumLabel(){
        return fetcher.totalOutcomes();
    }

}


