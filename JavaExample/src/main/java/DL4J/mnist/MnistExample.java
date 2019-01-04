package DL4J.mnist;

import com.sun.corba.se.spi.presentation.rmi.IDLNameTranslator;
import javafx.util.Pair;
import org.apache.avro.ipc.trace.ID;
import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.omg.PortableInterceptor.INACTIVE;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Author: songjun
 * Date: 2019/1/4
 * Description:
 **/
public class MnistExample {
    private static Logger logger = LoggerFactory.getLogger(MnistExample.class);
    public static void main(String[] args) throws IOException {
        int numEpoch = 20;
        int batchSize = 100;
        double learningRate = 0.001;

        DataSetIterator = new MnistDataSetIterExample(batchSize, )


        Pair<INDArray, INDArray> train_data = getMnistTrainData();
        INDArray train_feas = train_data.getKey();
        INDArray train_labels = train_data.getValue();
        long numSample = train_feas.shape()[0];
        long feaSize = train_feas.shape()[1];
        System.out.println(String.format("train feature shape: %d, %d", train_feas.shape()[0], train_feas.shape()[1]));
        System.out.println(String.format("train label shape: %d, %d", train_labels.shape()[0], train_labels.shape()[1]));

        Pair<INDArray, INDArray> eval_data = getMnistEvalData();
        INDArray eval_feas = eval_data.getKey();
        INDArray eval_labels = eval_data.getValue();
        long numLabel = eval_feas.shape()[1];
        System.out.println(String.format("eval feature shape: %d, %d", eval_feas.shape()[0], eval_feas.shape()[1]));
        System.out.println(String.format("eval label shape: %d, %d", eval_labels.shape()[1], eval_feas.shape()[1]));


        logger.info("Build model......");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate, 0.98))
                .l2(learningRate * 0.005)
                .list()
                .layer(0, new DenseLayer.Builder()
                    .nIn(feaSize)
                    .nOut(200)
                    .build())
                .layer(1, new DenseLayer.Builder()
                    .nIn(200)
                    .nOut(100)
                    .activation(Activation.RELU)
                    .build())
                .layer(2, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                    .activation(Activation.SOFTMAX)
                    .nIn(100)
                    .nOut(numLabel)
                    .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(5));

        logger.info("training model....");
        List<Long> order = new ArrayList<>();
        for(long i=0; i<numSample; ++i){
            order.add(i);
        }

        for(int i=0; i < numEpoch; ++i){
            long iterSampleNum = 0;
            while(iterSampleNum < numSample){
                iterSampleNum += batchSize;
                Collections.shuffle(order);
                List<Long> idxs = order.subList(0, batchSize);
                List<INDArray> batchFea = new ArrayList<>();
                List<INDArray> batchLabel = new ArrayList<>();
                for(Long idx : idxs){
                    batchFea.add(train_feas.get(NDArrayIndex.point(idx), NDArrayIndex.all()));
                    batchLabel.add(train_labels.get(NDArrayIndex.point(idx), NDArrayIndex.all()));
                }
                INDArray trFeas = Nd4j.vstack(batchFea);
                INDArray trLabel = Nd4j.vstack(batchLabel);
            }
            model.fit();
        }
    }
}
