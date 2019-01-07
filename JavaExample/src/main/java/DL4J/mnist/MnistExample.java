package DL4J.mnist;

import com.sun.corba.se.spi.presentation.rmi.IDLNameTranslator;
import javafx.util.Pair;
import org.apache.avro.ipc.trace.ID;
import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
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
        logger.info("start mnist example.");
        int numEpoch = 20;
        int batchSize = 100;
        double learningRate = 0.001;

        String train_data_path = "data/mnist/train_data.txt";
        String train_label_path = "data/mnist/train_labels.txt";
        MnistDataSetIterExample trainDataIter = new MnistDataSetIterExample(batchSize, new MnistDataFetcherExample(train_data_path, train_label_path));
        long feaSize = trainDataIter.getFeaSize();
        long numSample = trainDataIter.getNumSample();
        long numLabel = trainDataIter.getNumLabel();
        System.out.println("load train data finish.");

        String eval_data_path = "data/mnist/eval_data.txt";
        String eval_label_path = "data/mnist/eval_labels.txt";
        MnistDataSetIterExample evalDataIter = new MnistDataSetIterExample(batchSize, new MnistDataFetcherExample(eval_data_path, eval_label_path));
        System.out.println("load eval data finish.");

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

        logger.info("training model......");
        List<Long> order = new ArrayList<>();
        for(long i=0; i<numEpoch; ++i){
            logger.info("Epoch " + i);
            model.fit(trainDataIter);
        }

        logger.info("Evaluate model......");
        Evaluation eval = new Evaluation(evalDataIter.getNumLabel());
        while(evalDataIter.hasNext()){
            DataSet next = evalDataIter.next();
            INDArray output = model.output(next.getFeatures());
            eval.eval(next.getLabels(), output);
        }

        logger.info(eval.stats());
        logger.info("mnist finish.");

    }
}
