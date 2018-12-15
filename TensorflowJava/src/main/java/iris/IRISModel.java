package iris;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.*;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.List;

/**
 * Author: songjun
 * Date: 2018/12/14
 * Description:
 **/
public class IRISModel {
    public static void main(String[] args) throws IOException {
        String model_dir = "C:\\Projects\\github\\ExampleBank\\tensorflow_example\\data\\iris\\output\\saved_model\\output";
        System.out.printf("load model from %s%n", model_dir);

        SavedModelBundle bundle = SavedModelBundle.load(model_dir, "train");
        Session sess = bundle.session();
        Tensor eval_fea = get_eval_fea();
        Tensor eval_label = get_eval_label();

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
    }

    private static Tensor get_eval_fea() throws IOException {
        String eval_fea_file_path = "C:\\Projects\\github\\ExampleBank\\tensorflow_example\\data\\iris\\eval_data.txt";
        File f = new File(eval_fea_file_path);
        InputStreamReader reader = new InputStreamReader(new FileInputStream(f));
        BufferedReader br = new BufferedReader(reader);
        ArrayList<ArrayList<Float>> feas = new ArrayList<>();
        String line = "";
        line = br.readLine();
        while (line != null){
            String[] vals = line.split(" ");
            ArrayList<Float> fea = new ArrayList<>();
            for(String v : vals){
                fea.add(Float.parseFloat(v));
            }
            feas.add(fea);
            line = br.readLine();
        }
        int sample_num = feas.size();
        int fea_size = feas.get(0).size();

        FloatBuffer buf = FloatBuffer.allocate(sample_num * fea_size);
        for(ArrayList<Float> fea : feas){
            for(Float v : fea){
                buf.put(v);
            }
        }
        buf.flip();
        long[] shape = new long[] {sample_num, fea_size};
        Tensor t = Tensor.create(shape, buf);
        return t;
    }

    private static Tensor get_eval_label() throws IOException{
        String label_file_path = "C:\\Projects\\github\\ExampleBank\\tensorflow_example\\data\\iris\\eval_labels.txt";
        File f = new File(label_file_path);
        InputStreamReader reader = new InputStreamReader(new FileInputStream(f));
        BufferedReader br = new BufferedReader(reader);
        ArrayList<Long> labels = new ArrayList<>();
        String line = "";
        line = br.readLine();
        while (line != null){
            labels.add(Long.parseLong(line));
            line = br.readLine();
        }
        int sample_num = labels.size();
        LongBuffer buf = LongBuffer.allocate(sample_num);
        for(Long v : labels){
            buf.put(v);
        }
        buf.flip();
        long[] shape = new long[] {sample_num};
        Tensor t = Tensor.create(shape, buf);
        return t;
    }
}
