package DL4J.mnist;

import javafx.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.fetcher.BaseDataFetcher;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MnistDataFetcherExample extends BaseDataFetcher {
    private long iterSampleNum = 0;
    private List<Long> order = null;
    private INDArray dataFeas = null;
    private INDArray dataLabels = null;

    public MnistDataFetcherExample(String feaDataPath, String labelDataPath) throws IOException {
        Pair<INDArray, INDArray> fea_label = getFeaLabelData(feaDataPath, labelDataPath);
        this.dataFeas = fea_label.getKey();
        this.dataLabels = fea_label.getValue();
    }

    public Pair<INDArray, INDArray> getMnistTrainData() throws IOException {
        String train_data_path = "./data/train_data.txt";
        String train_label_path = "./data/train_labels.txt";
        return getFeaLabelData(train_data_path, train_label_path);
    }

    public Pair<INDArray, INDArray> getMnistEvalData() throws IOException {
        String eval_data_path = "./data/eval_data.txt";
        String eval_label_path = "./data/eval_labels.txt";
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
        int[] fea_shape = {fea_data.size(), (int)fea_data.get(0).size(0)};
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
        int[] label_shape = {labels_data.size(), (int)labels_data.get(0).size(0)};
        INDArray res_label = Nd4j.create(labels_data, label_shape, 'c');

        return new Pair<>(res_feas, res_label);
    }

    @Override
    public void fetch(int numSample) {

    }
}
