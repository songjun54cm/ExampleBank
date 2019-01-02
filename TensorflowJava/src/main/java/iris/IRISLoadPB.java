package iris;
import org.tensorflow.*;

import java.io.File;
import java.io.IOException;
import java.nio.LongBuffer;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
/**
 * Author: songjun
 * Date: 2018/12/15
 * Description:
 **/
public class IRISLoadPB {
    public static void main(String[] args) throws IOException {
        File modelFile = new File("..\\tensorflow_example\\data\\iris\\output\\frozen\\iris.pb");
        byte[] graphDef = Files.readAllBytes(modelFile.toPath());
        Graph graph = new Graph();
        graph.importGraphDef(graphDef);

        Tensor eval_fea = IRISLoadSavedModelBuilder.get_eval_fea();
        Tensor eval_label = IRISLoadSavedModelBuilder.get_eval_label();

        try(Session sess = new Session(graph)){
            List<Tensor<?>> res = sess.runner()
                    .feed("input_fea", eval_fea)
                    .feed("input_label", eval_label)
                    .fetch("pred_accuracy")
                    .fetch("pred_class")
                    .run();

            Tensor acc = res.get(0);
            Tensor pred_class = res.get(1);

            System.out.printf("predict accuracy: %f%n", acc.floatValue());
            System.out.printf("predict class shape %s%n", pred_class.shape().toString());
            System.out.printf("predict class %s%n", pred_class.toString());
            System.out.printf("num elmt: %d%n", pred_class.numElements());
            LongBuffer pred_cs = LongBuffer.allocate(pred_class.numElements());
            pred_class.writeTo(pred_cs);
            List<Long> pcs = new ArrayList<>();
            for(int i=0;i<pred_class.numElements(); ++i){
                pcs.add(pred_cs.get(i));
            }
            System.out.printf("pred class: %s%n", pcs.toString());

            for(Tensor t : res){
                t.close();
            }
        }
        eval_fea.close();
        eval_label.close();
    }
}
