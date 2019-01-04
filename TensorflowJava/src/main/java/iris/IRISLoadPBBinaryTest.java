package iris;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.*;
import java.nio.LongBuffer;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;

/**
 * Author: songjun
 * Date: 2018/12/15
 * Description:
 **/
public class IRISLoadPBBinaryTest {
    public static void main(String[] args) throws IOException {
        String path = saveModelToStr();

//        String path = "..\\tensorflow_example\\data\\iris\\output\\frozen\\iris_py.str";
//        String path = "..\\tensorflow_example\\data\\iris\\output\\frozen\\iris0.str";

        File modelStrFile = new File(path);
        InputStreamReader reader = new InputStreamReader(new FileInputStream(modelStrFile));
        BufferedReader br = new BufferedReader(reader);
        String modelStr = br.readLine();

        byte[] decodeBytes = Base64.getDecoder().decode(modelStr);

        Graph graph = new Graph();
        graph.importGraphDef(decodeBytes);

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

    public static String saveModelToStr() throws IOException {
        File modelFile = new File("..\\tensorflow_example\\data\\iris\\output\\frozen\\iris.pb");
        byte[] graphDef = Files.readAllBytes(modelFile.toPath());

        String encodeString = Base64.getEncoder().encodeToString(graphDef);
        System.out.printf("encoded string value: %n%s%n", encodeString);
        String resFilePath = "..\\tensorflow_example\\data\\iris\\output\\frozen\\iris.str";
        File modelStringFile = new File(resFilePath);
        modelStringFile.createNewFile();
        BufferedWriter out = new BufferedWriter(new FileWriter(modelStringFile));
        out.write(encodeString);
        out.flush();
        out.close();
        return resFilePath;
    }
}
