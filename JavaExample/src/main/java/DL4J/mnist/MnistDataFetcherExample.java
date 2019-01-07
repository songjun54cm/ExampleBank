package DL4J.mnist;

import javafx.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.fetcher.BaseDataFetcher;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class MnistDataFetcherExample extends BaseDataFetcher {
    private List<Long> order = null;
    private INDArray dataFeas = null;
    private INDArray dataLabels = null;

    public MnistDataFetcherExample(String feaDataPath, String labelDataPath) throws IOException {
        Pair<INDArray, INDArray> fea_label = getFeaLabelData(feaDataPath, labelDataPath);
        this.dataFeas = fea_label.getKey();
        this.dataLabels = fea_label.getValue();
        this.order = new ArrayList<>();
        for(int i=0; i<this.dataFeas.shape()[0]; ++i){
            this.order.add((long)i);
        }
        this.totalExamples = (int)this.dataFeas.shape()[0];
        this.numOutcomes = (int)this.dataLabels.shape()[1];
        this.inputColumns = (int) this.dataFeas.shape()[1];

    }
    @Override
    public void fetch(int numSample) {
        if (!this.hasMore()) {
            throw new IllegalStateException("Unable to get more; there are no more images");
        }else{
            List<Long> idxs = order.subList(this.cursor, Math.min(this.cursor+numSample, this.totalExamples));
            this.cursor += numSample;
            List<INDArray> curFeas = new ArrayList<>();
            List<INDArray> curLabels = new ArrayList<>();
            for(long idx : idxs){
                curFeas.add(this.dataFeas.get(NDArrayIndex.point(idx), NDArrayIndex.all()));
                curLabels.add(this.dataLabels.get(NDArrayIndex.point(idx), NDArrayIndex.all()));
            }

            int[] fea_shape = {curFeas.size(), (int)curFeas.get(0).shape()[1]};
            INDArray features = Nd4j.create(curFeas, fea_shape, 'c');
            int[] labelShape = {curLabels.size(), (int)curLabels.get(0).shape()[1]};
            INDArray labels = Nd4j.create(curLabels, labelShape, 'c');
            this.curr = new DataSet(features, labels);
        }
    }

    public void reset() {
        this.cursor = 0;
        Collections.shuffle(this.order);
    }

    public Pair<INDArray, INDArray> getMnistTrainData() throws IOException {
        String train_data_path = "../data/train_data.txt";
        String train_label_path = "../data/train_labels.txt";
        return getFeaLabelData(train_data_path, train_label_path);
    }

    public Pair<INDArray, INDArray> getMnistEvalData() throws IOException {
        String eval_data_path = "../data/eval_data.txt";
        String eval_label_path = "../data/eval_labels.txt";
        return getFeaLabelData(eval_data_path, eval_label_path);
    }

    public Pair<INDArray, INDArray> getFeaLabelData(String feaDataPath, String labelDataPath) throws IOException {
        File data_file = new File(feaDataPath);
        InputStreamReader reader = new InputStreamReader(new FileInputStream(data_file));
        ArrayList<INDArray> fea_data = new ArrayList<>();
        BufferedReader fea_br = new BufferedReader(reader);
        String line = "";
        line = fea_br.readLine();
        while(line != null){
            String[] vals = line.split(" ");
            float[] fea_vals = new float[vals.length];
            int i = 0;
            for(String v : vals){
                fea_vals[i] = Float.valueOf(v);
                i += 1;
            }
            INDArray feas = Nd4j.create(fea_vals);
            fea_data.add(feas);
            line = fea_br.readLine();
        }
        int[] fea_shape = {fea_data.size(), (int)fea_data.get(0).shape()[1]};
        INDArray res_feas = Nd4j.create(fea_data, fea_shape, 'c');

        File label_file = new File(labelDataPath);
        InputStreamReader label_reader = new InputStreamReader(new FileInputStream(label_file));
        ArrayList<INDArray> labels_data = new ArrayList<>();
        BufferedReader label_br = new BufferedReader(label_reader);
        String label_line = "";
        label_line = label_br.readLine();
        while(label_line != null){
            Integer label_v = Integer.valueOf(label_line);
            float[] label_vals = new float[10];
            Arrays.fill(label_vals, 0.0f);
            label_vals[label_v] = 1.0f;
            INDArray labels = Nd4j.create(label_vals);
            labels_data.add(labels);
            label_line = label_br.readLine();
        }
        int[] label_shape = {labels_data.size(), (int)labels_data.get(0).shape()[1]};
        INDArray res_label = Nd4j.create(labels_data, label_shape, 'c');

        long numSample = res_feas.shape()[0];
        long feaSize = res_label.shape()[1];
        System.out.println(String.format("feature shape: %d, %d", res_feas.shape()[0], res_feas.shape()[1]));
        System.out.println(String.format("label shape: %d, %d", res_label.shape()[0], res_label.shape()[1]));

        return new Pair<>(res_feas, res_label);
    }
}
