import org.apache.spark.ml.classification.LogisticRegressionModel;

public class TrySparkModel {
    public static void main(String[] args){
        LogisticRegressionModel lrModel = LogisticRegressionModel.load("./data/testLR.model");
    }
}
