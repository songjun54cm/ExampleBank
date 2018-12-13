package SimpleExample;

import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
public class LRND4JModel {
    public static void main(String[] args) throws IOException, ParseException {
        String modelFilePath = "data/testLRModel.txt";
        BufferedReader br = null;
        FileReader fr = null;

        fr = new FileReader(modelFilePath);
        br = new BufferedReader(fr);
        String configLine = br.readLine();

        JSONParser p = new JSONParser();
        JSONObject conf = (JSONObject)p.parse(configLine);

        String coefStr = conf.get("coefficients").toString();
        int spos = 0;
        int epos = coefStr.length();
        if(coefStr.startsWith("[")) spos = 1;
        if(coefStr.endsWith("]")) epos = epos-1;
        String[] vals = coefStr.substring(spos, epos).split(",");
        INDArray coeficients = Nd4j.zeros(vals.length+1);
        for(int i=0; i<vals.length; ++i){
            coeficients.putScalar(i, Double.valueOf(vals[i]));
        }
        printShape(coeficients);


        INDArray t1 = Nd4j.create(new double[]{-1.0, 1.5, 1.3, 0.0});
        INDArray t2 = Nd4j.create(new double[]{3.0, 2.0, -0.1, 0.0});
        INDArray t3 = Nd4j.create(new double[]{0.0, 2.2, -1.5, 0.0});
        INDArray test = Nd4j.vstack(t1, t2, t3);
        printShape(test);

//        INDArray test = Nd4j.create(new double[][]{{-1.0, 1.5, 1.3}, {3.0, 2.0, -0.1}, {0.0, 2.2, -1.5}});

        INDArray ndv = coeficients.mmul(test.transpose());
        System.out.println(ndv);


    }

    public static void printShape(INDArray array){
        int[] shape = array.shape();
        System.out.print("[");
        for(int i: shape){
            System.out.print(i);
            System.out.print(",");
        }
        System.out.println("]");
    }
}
