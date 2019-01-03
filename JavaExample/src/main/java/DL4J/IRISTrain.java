package DL4J;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Author: songjun
 * Date: 2018/12/13
 * Description:
 **/
public class IRISTrain {
    static int batch_size = 100;
    static int feaSize = 4;
    static int reportInterval = 5;
    static int labelNum = 3;

    public static void main(String... args) throws java.io.IOException {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().
                seed(1234).
                optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).
                updater(new Adam()).
                l2(1e-4).
                list().
                layer(new DenseLayer.Builder()
                    .nIn(feaSize)
                    .nOut(10)
                    .activation(Activation.SIGMOID)
                    .weightInit(WeightInit.NORMAL)
                    .build())
                .layer(new DenseLayer.Builder()
                    .nIn(10)
                    .nOut(feaSize)
                    .activation(Activation.SIGMOID)
                    .weightInit(WeightInit.NORMAL)
                    .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nIn(feaSize)
                    .nOut(labelNum)
                    .activation(Activation.SOFTMAX)
                    .weightInit(WeightInit.NORMAL)
                    .build())
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork dnn = new MultiLayerNetwork(conf);
        dnn.init();

    }
}
