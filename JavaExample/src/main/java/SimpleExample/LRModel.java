package SimpleExample;

import org.apache.commons.collections.iterators.ArrayListIterator;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class LRModel {
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
        List<Double> coefs = new ArrayList<Double>();
        for(String v: vals){
            coefs.add(Double.valueOf(v));
        }

        List<List<Double>> test = new ArrayList<List<Double>>();
        test.add(Arrays.asList(-1.0, 1.5, 1.3, 0.0));
        test.add(Arrays.asList(3.0, 2.0, -0.1, 0.0));
        test.add(Arrays.asList(0.0, 2.2, -1.5, 0.0));

        List<Double> scores = new ArrayList<Double>();
        for(List<Double> fea:test){
            scores.add(lrPredict(coefs, fea));
        }
        System.out.println(scores.toString());
    }

    private static double lrPredict(List<Double> coefs, List<Double> feas){
        double v = 0.0;
        for(int i=0; i<coefs.size(); ++i){
            v += coefs.get(i) * feas.get(i);
        }
        return v;
    }
}
